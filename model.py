import torch
import torch.nn as nn
from backbone import ResNet,Res2Net
from neck import FPN,SEPC,SEPC_Decoder
from head import Decoder
import torch.nn.functional as F
class salSEPCema(nn.Module):
    def __init__(self,alpha,backbone_name,pretrained=(
        'https://github.com/shinya7y/UniverseNet/releases/download/20.06/'
        'res2net50_v1b_26w_4s-3cf99910_mmdetv2.pth')):
        super(salSEPCema,self).__init__()
        backbone_cfg=dict(depth=50,deep_stem=True,avg_down=True,frozen_stages=1,dcn={'type':'DCN','deform_groups':1,'fallback_on_stride':False},
                          stage_with_dcn=(False, True, True, True))
        fpn_cfg=dict(in_channels=[256,512,1024,2048],out_channels=256,num_outs=5,start_level=0,add_extra_convs='on_output')
        sepc_cfg=dict(pconv_deform=True,lcconv_deform=True,lcconv_padding=1)
        self.backbone=Res2Net(**backbone_cfg)
        self.fpn=FPN(**fpn_cfg)
        self.sepc=SEPC_Decoder(**sepc_cfg)
        if alpha == None:
            self.alpha = nn.Parameter(torch.Tensor([0.25]))
            print("Initial alpha set to: {}".format(self.alpha))
        self.semantic_branch = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)
        self.output=nn.Conv2d(128, 1, kernel_size=3, stride=1, padding=1)
        # self.decoder=Decoder()
        self.gn1 = nn.GroupNorm(128, 128)

        self.backbone.init_weights(pretrained=pretrained)

    def forward(self,x,prev_state=None):
        # x=x.squeeze()
        x=self.backbone(x)
        x=self.fpn(x)
        if prev_state is None:
            current_state = x[1]
        else:
            current_state = torch.sigmoid(self.alpha) * x[1] + (1 - torch.sigmoid(self.alpha)) * prev_state
        x=list(x)
        x[1]=current_state
        x=self.sepc(x)
        x=F.relu(self.gn1(self.semantic_branch(x[0]+x[1]+x[2]+x[3]+x[4])))
        x=self.output(x)
        # x=self.decoder(x)
        x = torch.sigmoid(x)
        return current_state,x

class salSEPC(nn.Module):
    def __init__(self,backbone_name,pretrained=(
        'https://github.com/shinya7y/UniverseNet/releases/download/20.06/'
        'res2net50_v1b_26w_4s-3cf99910_mmdetv2.pth')):
        super(salSEPC,self).__init__()
        backbone_cfg=dict(depth=50,deep_stem=True,avg_down=True,frozen_stages=1,dcn={'type':'DCN','deform_groups':1,'fallback_on_stride':False},
                          stage_with_dcn=(False, True, True, True))
        fpn_cfg=dict(in_channels=[256,512,1024,2048],out_channels=256,num_outs=5,start_level=0,add_extra_convs='on_output')
        sepc_cfg=dict(pconv_deform=True,lcconv_deform=True,lcconv_padding=1)
        self.backbone=Res2Net(**backbone_cfg)
        self.fpn=FPN(**fpn_cfg)
        self.sepc=SEPC_Decoder(**sepc_cfg)
        self.semantic_branch = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)
        self.output=nn.Conv2d(128, 1, kernel_size=3, stride=1, padding=1)
        # self.decoder=Decoder()
        self.gn1 = nn.GroupNorm(128, 128)

        self.backbone.init_weights(pretrained=pretrained)

    def forward(self,x):
        # x=x.squeeze()
        x=self.backbone(x)
        x=self.fpn(x)
        x=self.sepc(x)
        x=F.relu(self.gn1(self.semantic_branch(x[0]+x[1]+x[2]+x[3]+x[4])))
        x=self.output(x)
        # x=self.decoder(x)
        x=F.sigmoid(x)
        return x

