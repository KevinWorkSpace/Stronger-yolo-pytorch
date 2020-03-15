from models.backbone import *
from models.backbone.helper import *
from models.backbone.baseblock import *
import utils.GIOU as GIOUloss


class BaseModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.numclass = cfg.numcls
        self.gt_per_grid = cfg.gt_per_grid
        self.backbone = eval(cfg.backbone)(pretrained=cfg.backbone_pretrained)
        self.outC = self.backbone.backbone_outchannels
        self.heads = []
        self.activate_type = 'relu6'
        self.input_size = 512
        self.bcelogit_loss = torch.nn.BCEWithLogitsLoss(reduction='none')
        self.smooth_loss = torch.nn.SmoothL1Loss(reduction='none')

    def get_grid(self, bz, gridsize):
        shiftx = torch.arange(0, gridsize, dtype=torch.float32)
        shifty = torch.arange(0, gridsize, dtype=torch.float32)
        shifty, shiftx = torch.meshgrid([shiftx, shifty])
        shiftx = shiftx.unsqueeze(-1).repeat(bz, 1, 1, self.gt_per_grid)
        shifty = shifty.unsqueeze(-1).repeat(bz, 1, 1, self.gt_per_grid)
        xy_grid = torch.stack([shiftx, shifty], dim=4).cuda()
        return xy_grid

    def decode(self, output, stride):
        bz = output.shape[0]
        gridsize = output.shape[-1]
        output = output.permute(0, 2, 3, 1)
        output = output.view(bz, gridsize, gridsize, self.gt_per_grid, 5 + self.numclass)
        x1y1, x2y2, conf, prob = torch.split(output, [2, 2, 1, self.numclass], dim=4)
        xy_grid = self.get_grid(bz, gridsize)
        x1y1 = (xy_grid + 0.5 - torch.exp(x1y1)) * stride
        x2y2 = (xy_grid + 0.5 + torch.exp(x2y2)) * stride

        xyxy = torch.cat((x1y1, x2y2), dim=4)
        output = torch.cat((xyxy, conf, prob), 4)
        output = output.view(bz, -1, 5 + self.numclass)
        return output

    def decode_infer(self, output, stride):
        bz = output.shape[0]
        gridsize = output.shape[-1]

        output = output.permute(0, 2, 3, 1)
        output = output.view(bz, gridsize, gridsize, self.gt_per_grid, 5 + self.numclass)
        x1y1, x2y2, conf, prob = torch.split(output, [2, 2, 1, self.numclass], dim=4)
        xy_grid = self.get_grid(bz, gridsize)

        x1y1 = (xy_grid + 0.5 - torch.exp(x1y1)) * stride
        x2y2 = (xy_grid + 0.5 + torch.exp(x2y2)) * stride

        xyxy = torch.cat((x1y1, x2y2), dim=4)
        conf = torch.sigmoid(conf)
        prob = torch.sigmoid(prob)
        output = torch.cat((xyxy, conf, prob), 4)
        output = output.view(bz, -1, 5 + self.numclass)
        return output

    def forward(self, input, targets=None):
        self.input_size = input.shape[-1]
        feat_small, feat_mid, feat_large = self.backbone(input)
        conv = self.headslarge(feat_large)
        convlarge = conv

        conv = self.mergelarge(convlarge)
        conv = self.headsmid(torch.cat((conv, feat_mid), dim=1))
        convmid = conv

        conv = self.mergemid(convmid)

        conv = self.headsmall(torch.cat((conv, feat_small), dim=1))
        convsmall = conv
        if self.cfg.ASFF:
            convlarge = self.asff0(convlarge, convmid, convsmall)
            convmid = self.asff1(convlarge, convmid, convsmall)
            convsmall = self.asff2(convlarge, convmid, convsmall)
        outlarge = self.detlarge(convlarge)
        outmid = self.detmid(convmid)
        outsmall = self.detsmall(convsmall)
        if self.training:
            assert targets is not None
            predlarge = self.decode(outlarge, 32)
            predmid = self.decode(outmid, 16)
            predsmall = self.decode(outsmall, 8)
            return self.loss([predsmall, predmid, predlarge], targets)
        else:
            predlarge = self.decode_infer(outlarge, 32)
            predmid = self.decode_infer(outmid, 16)
            predsmall = self.decode_infer(outsmall, 8)
            pred = torch.cat([predsmall, predmid, predlarge], dim=1)
            return pred

    def loss(self, preds: list, gtbox: list):
        """
        :param preds: [feat1,feat2,feat3]->[bz,pointnum,5+self.numclass]
        :param gtbox: gtbox and label for each image, [bz,[N,5]]
        :return:
        """

        def focalloss(target, actual, alpha=1, gamma=2):
            focal = alpha * torch.pow(torch.abs(target - actual), gamma)
            return focal

        cls_reg_targets = self.build_target(gtbox, preds)
        preds = torch.cat(preds, dim=1)

        pred_coor = preds[..., 0:4]
        pred_prob = preds[..., 5:]
        pred_prob_decode = torch.sigmoid(pred_prob)
        pred_conf = preds[..., 4:5]
        pred_conf_decode = torch.sigmoid(pred_conf)

        label_coor = cls_reg_targets[..., 0:4]
        respond_bbox = cls_reg_targets[..., 4:5]
        respond_bgd = cls_reg_targets[..., -1][..., None]
        label_mixw = cls_reg_targets[..., 5:6]
        label_prob = cls_reg_targets[..., 6:-1]
        # 计算GIOU损失
        bbox_wh = label_coor[..., 2:] - label_coor[..., :2]
        bbox_loss_scale = 2.0 - 1.0 * bbox_wh[..., 0:1] * bbox_wh[..., 1:2] / (self.input_size ** 2)
        if self.cfg.boxloss == 'iou':
            giou = GIOUloss.GIOU(pred_coor, label_coor).unsqueeze(-1)
            giou_loss = respond_bbox * bbox_loss_scale * (1.0 - giou)
            bbox_loss = giou_loss
        elif self.cfg.boxloss == 'l1':
            l1_loss = respond_bbox * bbox_loss_scale * self.smooth_loss(target=label_coor,
                                                                        input=pred_coor) * self.cfg.l1scale
            bbox_loss = l1_loss
        elif self.cfg.boxloss == 'KL':
            raise NotImplementedError("See strongerv3kl.py")
            # l1_loss = respond_bbox * bbox_loss_scale * (
            #         torch.exp(-pred_vari) * self.smooth_loss(target=label_coor,
            #                                             input=pred_coor) + 0.5 * pred_vari) * self.cfg.l1scale
            # bbox_loss = l1_loss
        elif self.cfg.boxloss == 'diou':
            diou = GIOUloss.DIOU(pred_coor, label_coor).unsqueeze(-1)
            diou_loss = respond_bbox * bbox_loss_scale * (1.0 - diou)
            bbox_loss = diou_loss
        else:
            raise NotImplementedError
        bbox_loss = bbox_loss * label_mixw
        # (2)计算confidence损失
        conf_focal = focalloss(respond_bbox, pred_conf_decode)

        conf_loss = conf_focal * (
                respond_bbox * self.bcelogit_loss(target=respond_bbox, input=pred_conf)
                +
                respond_bgd * self.bcelogit_loss(target=respond_bbox, input=pred_conf)
        )
        conf_loss = conf_loss * label_mixw
        # (3)计算classes损失
        if pred_prob_decode.shape[-1] != 0:
            if self.cfg.clsfocal:
                cls_focal = focalloss(label_prob, pred_prob_decode)
                prob_loss = cls_focal * respond_bbox * self.bcelogit_loss(target=label_prob, input=pred_prob)
            else:
                prob_loss = respond_bbox * self.bcelogit_loss(target=label_prob, input=pred_prob)
        else:
            prob_loss = torch.zeros_like(label_prob)
        prob_loss = prob_loss * label_mixw
        return bbox_loss, conf_loss, prob_loss

    def build_target(self, bboxs: list, preds):
        raise NotImplementedError



if __name__ == '__main__':
    import torch.onnx

    # net=YoloV3(20)
    net = YoloV3(0)
    load_tf_weights(net, 'cocoweights-half.pkl')

    assert 0
    model = net.eval()
    load_checkpoint(model, torch.load('checkpoints/coco512_prune/checkpoint-best.pth'))
    statedict = model.state_dict()
    layer2block = defaultdict(list)
    for k, v in model.state_dict().items():
        if 'num_batches_tracked' in k:
            statedict.pop(k)
    for idx, (k, v) in enumerate(statedict.items()):
        if 'mobilev2' in k:
            continue
        else:
            flag = k.split('.')[1]
            layer2block[flag].append((k, v))
    for k, v in layer2block.items():
        print(k, len(v))
    pruneratio = 0.1

    # #onnx
    # input = torch.randn(1, 3, 320, 320)
    # torch.onnx.export(net, input, "coco320.onnx", verbose=True)
    # #onnxcheck
    # model=onnx.load("coco320.onnx")
    # onnx.checker.check_model(model)
