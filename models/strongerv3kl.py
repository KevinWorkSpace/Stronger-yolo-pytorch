from models.backbone import *
from models.backbone.helper import *
from models.backbone.baseblock import *
import utils.GIOU as GIOUloss
from models.BaseModel import BaseModel


class StrongerV3KL(BaseModel):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.activate_type = 'relu6'
        self.headslarge = nn.Sequential(OrderedDict([
            ('conv0', conv_bn(self.outC[0], 512, kernel=1, stride=1, padding=0)),
            ('conv1', sepconv_bn(512, 1024, kernel=3, stride=1, padding=1, seprelu=cfg.seprelu)),
            ('conv2', conv_bn(1024, 512, kernel=1, stride=1, padding=0)),
            ('conv3', sepconv_bn(512, 1024, kernel=3, stride=1, padding=1, seprelu=cfg.seprelu)),
            ('conv4', conv_bn(1024, 512, kernel=1, stride=1, padding=0)),
        ]))
        self.detlarge = nn.Sequential(OrderedDict([
            ('conv5', sepconv_bn(512, 1024, kernel=3, stride=1, padding=1, seprelu=cfg.seprelu)),
            ('conv6', conv_bias(1024, self.gt_per_grid * (self.numclass + 5 + 4), kernel=1, stride=1, padding=0))
        ]))
        self.mergelarge = nn.Sequential(OrderedDict([
            ('conv7', conv_bn(512, 256, kernel=1, stride=1, padding=0)),
            ('upsample0', nn.UpsamplingNearest2d(scale_factor=2)),
        ]))
        # -----------------------------------------------
        self.headsmid = nn.Sequential(OrderedDict([
            ('conv8', conv_bn(self.outC[1] + 256, 256, kernel=1, stride=1, padding=0)),
            ('conv9', sepconv_bn(256, 512, kernel=3, stride=1, padding=1, seprelu=cfg.seprelu)),
            ('conv10', conv_bn(512, 256, kernel=1, stride=1, padding=0)),
            ('conv11', sepconv_bn(256, 512, kernel=3, stride=1, padding=1, seprelu=cfg.seprelu)),
            ('conv12', conv_bn(512, 256, kernel=1, stride=1, padding=0)),
        ]))
        self.detmid = nn.Sequential(OrderedDict([
            ('conv13', sepconv_bn(256, 512, kernel=3, stride=1, padding=1, seprelu=cfg.seprelu)),
            ('conv14', conv_bias(512, self.gt_per_grid * (self.numclass + 5 + 4), kernel=1, stride=1, padding=0))
        ]))
        self.mergemid = nn.Sequential(OrderedDict([
            ('conv15', conv_bn(256, 128, kernel=1, stride=1, padding=0)),
            ('upsample0', nn.UpsamplingNearest2d(scale_factor=2)),
        ]))
        # -----------------------------------------------
        self.headsmall = nn.Sequential(OrderedDict([
            ('conv16', conv_bn(self.outC[2] + 128, 128, kernel=1, stride=1, padding=0)),
            ('conv17', sepconv_bn(128, 256, kernel=3, stride=1, padding=1, seprelu=cfg.seprelu)),
            ('conv18', conv_bn(256, 128, kernel=1, stride=1, padding=0)),
            ('conv19', sepconv_bn(128, 256, kernel=3, stride=1, padding=1, seprelu=cfg.seprelu)),
            ('conv20', conv_bn(256, 128, kernel=1, stride=1, padding=0)),
        ]))
        self.detsmall = nn.Sequential(OrderedDict([
            ('conv21', sepconv_bn(128, 256, kernel=3, stride=1, padding=1, seprelu=cfg.seprelu)),
            ('conv22', conv_bias(256, self.gt_per_grid * (self.numclass + 5 + 4), kernel=1, stride=1, padding=0))
        ]))
        if cfg.ASFF:
            self.asff0 = ASFF(0, activate=self.activate_type)
            self.asff1 = ASFF(1, activate=self.activate_type)
            self.asff2 = ASFF(2, activate=self.activate_type)

    def decode(self, output, stride):
        bz = output.shape[0]
        gridsize = output.shape[-1]

        output = output.permute(0, 2, 3, 1)
        output = output.view(bz, gridsize, gridsize, self.gt_per_grid, 5 + self.numclass + 4)
        x1y1, x2y2, variance, conf, prob = torch.split(output, [2, 2, 4, 1, self.numclass], dim=4)

        xy_grid = self.get_grid(bz, gridsize)

        x1y1 = (xy_grid + 0.5 - torch.exp(x1y1)) * stride
        x2y2 = (xy_grid + 0.5 + torch.exp(x2y2)) * stride

        xyxy = torch.cat((x1y1, x2y2), dim=4)
        output = torch.cat((xyxy, variance, conf, prob), 4)
        output = output.view(bz, -1, 9 + self.numclass)
        return output

    def decode_infer(self, output, stride):
        bz = output.shape[0]
        gridsize = output.shape[-1]

        output = output.permute(0, 2, 3, 1)
        output = output.view(bz, gridsize, gridsize, self.gt_per_grid, 5 + self.numclass + 4)
        x1y1, x2y2, variance, conf, prob = torch.split(output, [2, 2, 4, 1, self.numclass], dim=4)

        xy_grid = self.get_grid(bz, gridsize)

        x1y1 = (xy_grid + 0.5 - torch.exp(x1y1)) * stride
        x2y2 = (xy_grid + 0.5 + torch.exp(x2y2)) * stride

        xyxy = torch.cat((x1y1, x2y2), dim=4)
        conf = torch.sigmoid(conf)
        prob = torch.sigmoid(prob)
        output = torch.cat((xyxy, variance, conf, prob), 4)

        output = output.view(bz, -1, 5 + self.numclass + 4)
        return output

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
        # get tensor from prediction
        pred_coor = preds[..., 0:4]
        pred_vari = preds[..., 4:8]
        pred_conf = preds[..., 8:9]
        pred_conf_decode = torch.sigmoid(pred_conf)
        pred_prob = preds[..., 9:]
        pred_prob_decode = torch.sigmoid(pred_prob)

        # get tensor from groundTruth
        label_coor = cls_reg_targets[..., 0:4]
        respond_bbox = cls_reg_targets[..., 4:5]
        respond_bgd = cls_reg_targets[..., -1][..., None]
        label_mixw = cls_reg_targets[..., 5:6]
        label_prob = cls_reg_targets[..., 6:-1]
        # 计算GIOU损失
        bbox_wh = label_coor[..., 2:] - label_coor[..., :2]
        bbox_loss_scale = 2.0 - 1.0 * bbox_wh[..., 0:1] * bbox_wh[..., 1:2] / (self.input_size ** 2)

        bbox_loss = respond_bbox * bbox_loss_scale * (
                torch.exp(-pred_vari) * self.smooth_loss(target=label_coor,
                                                         input=pred_coor) + 0.5 * pred_vari) * self.cfg.l1scale

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
        # get target for each image
        batch_targets = []
        batch_preds = []
        for idx_img in range(preds[0].shape[0]):
            batch_preds.append(torch.cat([p[idx_img] for p in preds], 0))
        for bbox, pred in zip(bboxs, batch_preds):
            batch_targets.append(self.yolo_target_single(bbox, pred))
        batch_targets = torch.stack(batch_targets, 0)
        return batch_targets

    def yolo_target_single(self, target: torch.Tensor, pred: torch.Tensor):

        bbox, class_label, mix_weight = target.split([4, 1, 1], dim=1)
        class_label = class_label.long()
        reg_area_limit = [0, 30, 90, 10000]
        strides = [8, 16, 32]
        target_lvl = [torch.zeros(self.input_size // strides[i], self.input_size // strides[i], self.gt_per_grid,
                                  6 + self.numclass).cuda() for i in range(3)]
        target_count = [torch.zeros(self.input_size // strides[i], self.input_size // strides[i]).long() for i in
                        range(3)]
        bbox_xywh = torch.cat([(bbox[:, 2:] + bbox[:, :2]) * 0.5,
                               bbox[:, 2:] - bbox[:, :2]], dim=-1)
        bboxarea = torch.sqrt(bbox_xywh[:, -2] * bbox_xywh[:, -1])
        for i in range(3):
            # initialize box weight 1
            target_lvl[i][:, :, :, 5] = 1.0
            valid_mask = (bboxarea > reg_area_limit[i]) & (bboxarea < reg_area_limit[i + 1])
            for gt_xywh, class_index, gt_xyxy, box_weight in zip(bbox_xywh[valid_mask], class_label[valid_mask],
                                                                 bbox[valid_mask], mix_weight[valid_mask]):
                gt_xywh = (gt_xywh / strides[i]).long()
                numgt = target_count[i][gt_xywh[1]][gt_xywh[0]]
                delta = 0.01
                if numgt == 0 and self.gt_per_grid > 1:
                    for n in range(self.gt_per_grid):
                        target_lvl[i][gt_xywh[1]][gt_xywh[0]][n][:4] = gt_xyxy
                        target_lvl[i][gt_xywh[1]][gt_xywh[0]][n][4] = 1.0
                        target_lvl[i][gt_xywh[1]][gt_xywh[0]][n][5] = box_weight
                        target_lvl[i][gt_xywh[1]][gt_xywh[0]][n][6:] = 1.0 / self.numclass * delta
                        target_lvl[i][gt_xywh[1]][gt_xywh[0]][n][6 + class_index] = (
                                                                                                1.0 - delta) + 1.0 / self.numclass * delta
                else:
                    target_lvl[i][gt_xywh[1]][gt_xywh[0]][numgt][:4] = gt_xyxy
                    target_lvl[i][gt_xywh[1]][gt_xywh[0]][numgt][4] = 1.0
                    target_lvl[i][gt_xywh[1]][gt_xywh[0]][numgt][5] = box_weight
                    target_lvl[i][gt_xywh[1]][gt_xywh[0]][numgt][6:] = 1.0 / self.numclass * delta
                    target_lvl[i][gt_xywh[1]][gt_xywh[0]][numgt][6 + class_index] = (
                                                                                                1.0 - delta) + 1.0 / self.numclass * delta
                target_count[i][gt_xywh[1]][gt_xywh[0]] = min(self.gt_per_grid - 1,
                                                              target_count[i][gt_xywh[1]][gt_xywh[0]] + 1)
        target_lvl = [t.view(-1, 6 + self.numclass) for t in target_lvl]
        target_lvl = torch.cat(target_lvl, 0)
        iou = GIOUloss.bbox_overlaps(pred[:, :4], target[:, :4])
        max_iou, _ = torch.max(iou, dim=-1)
        max_iou = max_iou.unsqueeze(-1)
        respond_bgd = (torch.ones_like(target_lvl[:, 4:5]) - target_lvl[:, 4:5]) * (max_iou < 0.5).float()
        # respond_bgd = (torch.ones_like(target_lvl[:,4:5])- target_lvl[:,4:5])
        target_lvl = torch.cat([target_lvl, respond_bgd], -1)
        return target_lvl


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
