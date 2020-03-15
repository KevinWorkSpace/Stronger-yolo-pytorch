import numpy as np
import random
import torch
import torch.utils.data as data

class BaseDataset(data.Dataset):
    def __init__(self, cfg, subset, istrain):
        self.dataset_root = cfg.DATASET.dataset_root
        self._ids = []
        self.testsize = cfg.EXPER.test_size
        self.batch_size = cfg.OPTIM.batch_size
        self.trainsizes = cfg.EXPER.train_sizes
        self.istrain = istrain
        self._gt_per_grid = cfg.MODEL.gt_per_grid
        self.strides = np.array([8, 16, 32])
        self.numcls = cfg.MODEL.numcls
        self.labels = cfg.MODEL.LABEL

    def __len__(self):
        raise NotImplementedError

    def _parse_annotation(self, itemidx, random_trainsize):
        raise NotImplementedError

    def _load_batch(self, idx_batch, random_trainsize):
        batch_image = []
        batch_box = []
        batch_mixweight = []
        imgpath_batch = []
        orishape_batch = []
        for idx in range(self.batch_size):
            idxitem = idx_batch * self.batch_size + idx
            image_org, bboxes_org, labels_org, imgpath, ori_shape = self._parse_annotation(idxitem, random_trainsize)
            if random.random() < 0.5 and self.istrain:
                index_mix = random.randint(0, len(self._ids) - 1)
                image_mix, bboxes_mix, label_mix, _, _ = self._parse_annotation(index_mix, random_trainsize)

                lam = np.random.beta(1.5, 1.5)
                img = lam * image_org + (1 - lam) * image_mix
                mixw_org = torch.ones(bboxes_org.shape[0]) * lam
                mixw_mix = torch.ones(bboxes_mix.shape[0]) * (1 - lam)
                mix_weight = torch.cat([mixw_org, mixw_mix])
                bboxes = np.concatenate([bboxes_org, bboxes_mix])
                labels = np.concatenate([labels_org,label_mix])
            else:
                img = image_org
                bboxes = bboxes_org
                labels=labels_org
                mix_weight = torch.ones(bboxes_org.shape[0]).float()
            batch_image.append(img)
            batch_mixweight.append(mix_weight)
            targets=np.concatenate([bboxes,labels[...,None],mix_weight[...,None]],1).astype(np.float32)
            batch_box.append(torch.from_numpy(targets))
            imgpath_batch.append(imgpath)
            orishape_batch.append(ori_shape)
        return torch.from_numpy(np.array(batch_image).transpose((0, 3, 1, 2)).astype(np.float32)), \
               imgpath_batch, \
               torch.from_numpy(np.array(orishape_batch).astype(np.float32)), \
               batch_box

    def __getitem__(self, item):
        if self.istrain:
            trainsize = random.choice(self.trainsizes)
        else:
            trainsize = self.testsize

        return self._load_batch(item, trainsize)


if __name__ == '__main__':
    from yacscfg import _C as cfg
    import os
    import argparse

    parser = argparse.ArgumentParser(description="DEMO configuration")
    parser.add_argument(
        "--config-file",
        default='configs/strongerv1.yaml'
    )

    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    args = parser.parse_args()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.EVAL.iou_thres = 0.5
    cfg.DATASET.dataset_root = '/disk3/datasets/VOCdevkit'
    cfg.freeze()
    train, val = get_dataset(cfg)
    for data in train:
        print(len(train))
        assert 0
