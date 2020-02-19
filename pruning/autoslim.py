from pruning.BasePruner import BasePruner
from pruning.Block import *
from models.backbone.baseblock import InvertedResidual, conv_bn, sepconv_bn, conv_bias,DarknetBlock
from models.backbone.baseblock_US import bn_calibration_init
from pruning.BasePruner import BasePruner
import torch
import numpy as np
import os
import torch
import torch.nn as nn
from pruning.Block import *
from models import *
from collections import OrderedDict
import time
from thop import clever_format
class AutoSlimPruner(BasePruner):
    def __init__(self,Trainer, newmodel, cfg):
        super().__init__(Trainer, newmodel, cfg)
        self.pruneratio = cfg.pruneratio
        self.prunestep = 8
        self.constrain = 3e9

    def prune(self):
        blocks = [None]
        name2layer = {}
        for midx, (name, module) in enumerate(self.model.named_modules()):
            if type(module) not in [InvertedResidual, conv_bn, nn.Linear, sepconv_bn, conv_bias, DarknetBlock]:
                continue
            idx = len(blocks)
            if isinstance(module, DarknetBlock):
                blocks.append(DarkBlock(name, idx, [blocks[-1]], list(module.state_dict().values())))
            if isinstance(module, InvertedResidual):
                blocks.append(InverRes(name, idx, [blocks[-1]], list(module.state_dict().values())))
            if isinstance(module, conv_bn):
                blocks.append(CB(name, idx, [blocks[-1]], list(module.state_dict().values())))
            if isinstance(module, nn.Linear):
                blocks.append(FC(name, idx, [blocks[-1]], list(module.state_dict().values())))
            if isinstance(module, sepconv_bn):
                blocks.append(DCB(name, idx, [blocks[-1]], list(module.state_dict().values())))
            if isinstance(module, conv_bias):
                blocks.append(Conv(name, idx, [blocks[-1]], list(module.state_dict().values())))
            name2layer[name] = blocks[-1]
        self.blocks = blocks[1:]
        for b in self.blocks:
            # 两个输入的层由head部分决定
            if b.layername == 'mergelarge.conv7':
                b.inputlayer = [name2layer['headslarge.conv4']]
                b.bnscale=None
            # if b.layername == 'headsmid.conv8':
            #     b.inputlayer.append(name2layer[self.args.bbOutName[1]])
            if b.layername == 'mergemid.conv15':
                b.inputlayer = [name2layer['headsmid.conv12']]
                b.bnscale = None
            # if b.layername == 'headsmall.conv16':
            #     b.inputlayer.append(name2layer[self.args.bbOutName[0]])
            # if b.layername in self.args.bbOutName[:2]:
            #     b.bnscale=None
        # gather BN weights
        block_channels = OrderedDict()
        for idx, b in enumerate(self.blocks):
            if b.bnscale is None:
                block_channels.update({idx: None})
            else:
                block_channels.update({
                    idx:
                        {'numch': b.bnscale.shape[0],
                         'flops': 0,
                         'params': 0, }
                })
                block_channels.update({
                    'map':0.0
                })
                b.prunemask = torch.arange(0, b.bnscale.shape[0])
        # for k,v in block_channels.items():
        #     print(self.blocks[k].layername,v)
        # assert 0
        prune_iter=0
        s=time.time()
        while(1):
            prune_results = []
            for idx, b in enumerate(self.blocks):
                # if idx<26:
                #     continue
                if (block_channels[idx] == None or (block_channels[idx]['numch'] - self.prunestep)<=0):
                    prune_results.append(-1)
                    continue
                b.prunemask = torch.arange(0, block_channels[idx]['numch'] - self.prunestep).cuda()
                # always keep the adding layer together
                # name2layer[self.args.bbOutName[1]].inputmask=name2layer['mergelarge.conv7'].prunemask
                # name2layer[self.args.bbOutName[0]].inputmask=name2layer['mergemid.conv15'].prunemask
                assert b.prunemask.shape[0]>0
                self.clone_model()
                flops, params = self.get_flops(self.newmodel)
                block_channels[idx]['flops'] = flops
                block_channels[idx]['params'] = params
                accpruned = self.test(validiter=200,newmodel=True, cal_bn=True)
                prune_results.append(accpruned)
                # reset prunemask
                b.prunemask = torch.arange(0, block_channels[idx]['numch']).cuda()
            pick_idx=prune_results.index(max(prune_results))
            if block_channels[pick_idx]['flops']<self.constrain:
                break
            block_channels[pick_idx]['numch']-=self.prunestep
            self.blocks[pick_idx].prunemask=torch.arange(0, block_channels[pick_idx]['numch']).cuda()
            print("iteration {}: prune {},current flops:{},current params:{} ,results:{},spend {}sec".format(
                prune_iter,pick_idx,block_channels[pick_idx]['flops'],block_channels[pick_idx]['params'],max(prune_results),round(time.time()-s)))
            block_channels['map']=max(prune_results)
            torch.save(block_channels,'logs/{}.pth'.format(prune_iter))
            prune_iter+=1
        bns = []
