import torch
import torch.nn as nn
from backbone import ResNet,Res2Net
from neck import FPN,SEPC,SEPC_Decoder
from head import Decoder
import torch.nn.functional as F
from torchvision.models import vgg16
from torch.nn.modules.upsampling import Upsample
from torch.nn.modules.conv import Conv2d
from torch.nn.modules.activation import Sigmoid, ReLU
from torch.nn.functional import interpolate, dropout2d
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


class Upsample(nn.Module):
    # Upsample has been deprecated, this workaround allows us to still use the function within sequential.https://discuss.pytorch.org/t/using-nn-function-interpolate-inside-nn-sequential/23588
    def __init__(self, scale_factor, mode):
        super(Upsample, self).__init__()
        self.interp = interpolate
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x):
        x = self.interp(x, scale_factor=self.scale_factor, mode=self.mode)
        return x



class SalEMA(nn.Module):
    """
    In this model, we pick a Convolutional layer from the bottleneck and apply EMA as a simple temporal regularizer.
    The smaller the alpha, the less each newly added frame will impact the outcome. This way the temporal information becomes most relevant.
    """

    def __init__(self, alpha, ema_loc, residual, dropout, use_gpu=True):
        super(SalEMA, self).__init__()

        self.dropout = dropout
        self.residual = residual
        self.use_gpu = use_gpu
        if alpha == None:
            self.alpha = nn.Parameter(torch.Tensor([0.25]))
            print("Initial alpha set to: {}".format(self.alpha))
        else:
            self.alpha = torch.Tensor([float(alpha)])
            if use_gpu:
                self.alpha = self.alpha.cuda()

        assert (self.alpha <= 1 and self.alpha >= 0)
        self.ema_loc = ema_loc  # 30 = bottleneck

        # Create encoder based on VGG16 architecture
        original_vgg16 = vgg16()

        # select only convolutional layers
        encoder = torch.nn.Sequential(*list(original_vgg16.features)[:30])

        # define decoder based on VGG16 (inverse order and Upsampling layers)
        decoder_list = [
            Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            ReLU(),
            Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            ReLU(),
            Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            ReLU(),
            Upsample(scale_factor=2, mode='nearest'),

            Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            ReLU(),
            Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            ReLU(),
            Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            ReLU(),
            Upsample(scale_factor=2, mode='nearest'),

            Conv2d(512, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            ReLU(),
            Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            ReLU(),
            Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            ReLU(),
            Upsample(scale_factor=2, mode='nearest'),

            Conv2d(256, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            ReLU(),
            Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            ReLU(),
            Upsample(scale_factor=2, mode='nearest'),

            Conv2d(128, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            ReLU(),
            Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            ReLU(),
            Conv2d(64, 1, kernel_size=(1, 1), stride=(1, 1), padding=0),
            Sigmoid(),
        ]

        decoder = torch.nn.Sequential(*decoder_list)

        # assamble the full architecture encoder-decoder
        self.salgan = torch.nn.Sequential(*(list(encoder.children()) + list(decoder.children())))



        print("Model initialized, EMA located at {}".format(self.salgan[self.ema_loc]))
        # print(len(self.salgan))

    def forward(self, input_, prev_state=None):
        x = self.salgan[:self.ema_loc](input_)
        residual = x
        batch_size = x.data.size()[0]
        spatial_size = x.data.size()[2:]

        if self.dropout == True:
            x = dropout2d(x)
        # salgan[self.ema_loc] will act as the temporal state
        if prev_state is None:
            current_state = self.salgan[self.ema_loc](
                x)  # Initially don't apply alpha as there is no prev state we will consistently have bad saliency maps at the start if we were to do so.
        else:
            current_state = torch.sigmoid(self.alpha) * self.salgan[self.ema_loc](x) + (1 - torch.sigmoid(self.alpha)) * prev_state

        if self.residual == True:
            x = current_state + residual
        else:
            x = current_state

        if self.ema_loc < len(self.salgan) - 1:
            x = self.salgan[self.ema_loc + 1:](x)

        return current_state, x  # x is a saliency map at this point