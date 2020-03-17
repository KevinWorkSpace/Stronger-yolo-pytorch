from models import *
from trainers import *
import json
from yacscfg import _C as cfg
import os
from torch import optim
import argparse
import torch
from thop import profile,clever_format
from utils.dist_util import *
import torch.multiprocessing as mp
import torch.distributed as dist
from utils.util import pick_avail_port
import random
import numpy as np
def seed_torch(seed=2018):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
def main(local_rank,ngpus_pernode,args,avail_port):
    seed_torch()
    dist.init_process_group(backend='nccl', init_method='tcp://127.0.0.1:{}'.format(avail_port), world_size=ngpus_pernode, rank=local_rank)
    torch.cuda.set_device(local_rank)
    net = eval(args.MODEL.modeltype)(cfg=args.MODEL).cuda(local_rank)
    net = torch.nn.parallel.DistributedDataParallel(net, device_ids=[local_rank])

    optimizer = optim.Adam(net.parameters(),lr=args.OPTIM.lr_initial)
    scheduler=optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.OPTIM.milestones, gamma=0.1)
    _Trainer = eval('Trainer_{}'.format(args.DATASET.dataset))(args=args,
                       model=net,
                       optimizer=optimizer,
                       lrscheduler=scheduler
                       )
    if args.do_test:
      _Trainer._valid_epoch(validiter=-1,verbose=True,cal_bn=False,width_mult=0.7)
    else:
      _Trainer.train()

## distrbuted tester for pruning
def main_worker(local_rank,ngpus_pernode,args,net,child_conn,avail_port,cal_bn=False,valid_iter=-1,verbose=True):
    dist.init_process_group(backend='nccl', init_method='tcp://127.0.0.1:{}'.format(avail_port), world_size=ngpus_pernode, rank=local_rank)
    torch.cuda.set_device(local_rank)
    net=net.cuda(local_rank)
    net = torch.nn.parallel.DistributedDataParallel(net, device_ids=[local_rank])
    optimizer = optim.Adam(net.parameters(),lr=args.OPTIM.lr_initial)
    scheduler=optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.OPTIM.milestones, gamma=0.1)
    _Trainer = eval('Trainer_{}'.format(args.DATASET.dataset))(args=args,
                       model=net,
                       optimizer=optimizer,
                       lrscheduler=scheduler
                       )
    if args.do_test:
      results,_=_Trainer._valid_epoch(validiter=valid_iter,cal_bn=cal_bn,verbose=verbose)
      synchronize()
      if is_main_process():
          child_conn.send(results)
          child_conn.close()
    else:
      _Trainer.train()
      results,_=_Trainer._valid_epoch(validiter=valid_iter,cal_bn=cal_bn,verbose=verbose)
      synchronize()
      if is_main_process():
          child_conn.send(results)
          child_conn.close()
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="DEMO configuration")
    parser.add_argument(
        "--config-file",
        default = 'configs/strongerv3_dist.yaml'
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
    cfg.freeze()
    # print(cfg)
    # seed_torch()
    port=pick_avail_port()
    mp.spawn(main,nprocs=cfg.ngpu,args=(cfg.ngpu,cfg,port))
