import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule, xavier_init
from mmcv.runner import auto_fp16

# from mmdet.models.builder import NECKS

class FPN(nn.Module):
    r"""Feature Pyramid Network.

    This is an implementation of paper `Feature Pyramid Networks for Object
    Detection <https://arxiv.org/abs/1612.03144>`_.

    Args:
        in_channels (List[int]): Number of input channels per scale.
        out_channels (int): Number of output channels (used at each scale)
        num_outs (int): Number of output scales.
        start_level (int): Index of the start input backbone level used to
            build the feature pyramid. Default: 0.
        end_level (int): Index of the end input backbone level (exclusive) to
            build the feature pyramid. Default: -1, which means the last level.
        add_extra_convs (bool | str): If bool, it decides whether to add conv
            layers on top of the original feature maps. Default to False.
            If True, its actual mode is specified by `extra_convs_on_inputs`.
            If str, it specifies the source feature map of the extra convs.
            Only the following options are allowed

            - 'on_input': Last feat map of neck inputs (i.e. backbone feature).
            - 'on_lateral':  Last feature map after lateral convs.
            - 'on_output': The last output feature map after fpn convs.
        extra_convs_on_inputs (bool, deprecated): Whether to apply extra convs
            on the original feature from the backbone. If True,
            it is equivalent to `add_extra_convs='on_input'`. If False, it is
            equivalent to set `add_extra_convs='on_output'`. Default to True.
        relu_before_extra_convs (bool): Whether to apply relu before the extra
            conv. Default: False.
        no_norm_on_lateral (bool): Whether to apply norm on lateral.
            Default: False.
        conv_cfg (dict): Config dict for convolution layer. Default: None.
        norm_cfg (dict): Config dict for normalization layer. Default: None.
        act_cfg (str): Config dict for activation layer in ConvModule.
            Default: None.
        upsample_cfg (dict): Config dict for interpolate layer.
            Default: `dict(mode='nearest')`

    Example:
        >>> import torch
        >>> in_channels = [2, 3, 5, 7]
        >>> scales = [340, 170, 84, 43]
        >>> inputs = [torch.rand(1, c, s, s)
        ...           for c, s in zip(in_channels, scales)]
        >>> self = FPN(in_channels, 11, len(in_channels)).eval()
        >>> outputs = self.forward(inputs)
        >>> for i in range(len(outputs)):
        ...     print(f'outputs[{i}].shape = {outputs[i].shape}')
        outputs[0].shape = torch.Size([1, 11, 340, 340])
        outputs[1].shape = torch.Size([1, 11, 170, 170])
        outputs[2].shape = torch.Size([1, 11, 84, 84])
        outputs[3].shape = torch.Size([1, 11, 43, 43])
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 num_outs,
                 start_level=0,
                 end_level=-1,
                 add_extra_convs=False,
                 extra_convs_on_inputs=True,
                 relu_before_extra_convs=False,
                 no_norm_on_lateral=False,
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=None,
                 upsample_cfg=dict(mode='nearest')):
        super(FPN, self).__init__()
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.relu_before_extra_convs = relu_before_extra_convs
        self.no_norm_on_lateral = no_norm_on_lateral
        self.fp16_enabled = False
        self.upsample_cfg = upsample_cfg.copy()

        if end_level == -1:
            self.backbone_end_level = self.num_ins
            assert num_outs >= self.num_ins - start_level
        else:
            # if end_level < inputs, no extra level is allowed
            self.backbone_end_level = end_level
            assert end_level <= len(in_channels)
            assert num_outs == end_level - start_level
        self.start_level = start_level
        self.end_level = end_level
        self.add_extra_convs = add_extra_convs
        assert isinstance(add_extra_convs, (str, bool))
        if isinstance(add_extra_convs, str):
            # Extra_convs_source choices: 'on_input', 'on_lateral', 'on_output'
            assert add_extra_convs in ('on_input', 'on_lateral', 'on_output')
        elif add_extra_convs:  # True
            if extra_convs_on_inputs:
                # For compatibility with previous release
                # TODO: deprecate `extra_convs_on_inputs`
                self.add_extra_convs = 'on_input'
            else:
                self.add_extra_convs = 'on_output'

        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()

        for i in range(self.start_level, self.backbone_end_level):
            l_conv = ConvModule(
                in_channels[i],
                out_channels,
                1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg if not self.no_norm_on_lateral else None,
                act_cfg=act_cfg,
                inplace=False)
            fpn_conv = ConvModule(
                out_channels,
                out_channels,
                3,
                padding=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                inplace=False)

            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)

        # add extra conv layers (e.g., RetinaNet)
        extra_levels = num_outs - self.backbone_end_level + self.start_level
        if self.add_extra_convs and extra_levels >= 1:
            for i in range(extra_levels):
                if i == 0 and self.add_extra_convs == 'on_input':
                    in_channels = self.in_channels[self.backbone_end_level - 1]
                else:
                    in_channels = out_channels
                extra_fpn_conv = ConvModule(
                    in_channels,
                    out_channels,
                    3,
                    stride=2,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    inplace=False)
                self.fpn_convs.append(extra_fpn_conv)

    # default init_weights for conv(msra) and norm in ConvModule
    def init_weights(self):
        """Initialize the weights of FPN module."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')

    @auto_fp16()
    def forward(self, inputs):
        """Forward function."""
        assert len(inputs) == len(self.in_channels)

        # build laterals
        laterals = [
            lateral_conv(inputs[i + self.start_level])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        # build top-down path
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            # In some cases, fixing `scale factor` (e.g. 2) is preferred, but
            #  it cannot co-exist with `size` in `F.interpolate`.
            if 'scale_factor' in self.upsample_cfg:
                laterals[i - 1] += F.interpolate(laterals[i],
                                                 **self.upsample_cfg)
            else:
                prev_shape = laterals[i - 1].shape[2:]
                laterals[i - 1] += F.interpolate(
                    laterals[i], size=prev_shape, **self.upsample_cfg)

        # build outputs
        # part 1: from original levels
        outs = [
            self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels)
        ]
        # part 2: add extra levels
        if self.num_outs > len(outs):
            # use max pool to get more levels on top of outputs
            # (e.g., Faster R-CNN, Mask R-CNN)
            if not self.add_extra_convs:
                for i in range(self.num_outs - used_backbone_levels):
                    outs.append(F.max_pool2d(outs[-1], 1, stride=2))
            # add conv layers on top of original feature maps (RetinaNet)
            else:
                if self.add_extra_convs == 'on_input':
                    extra_source = inputs[self.backbone_end_level - 1]
                elif self.add_extra_convs == 'on_lateral':
                    extra_source = laterals[-1]
                elif self.add_extra_convs == 'on_output':
                    extra_source = outs[-1]
                else:
                    raise NotImplementedError
                outs.append(self.fpn_convs[used_backbone_levels](extra_source))
                for i in range(used_backbone_levels + 1, self.num_outs):
                    if self.relu_before_extra_convs:
                        outs.append(self.fpn_convs[i](F.relu(outs[-1])))
                    else:
                        outs.append(self.fpn_convs[i](outs[-1]))
        return tuple(outs)




import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import build_norm_layer
from torch.nn.modules.batchnorm import _BatchNorm

from mmdet.core import auto_fp16
from mmdet.ops.dcn.sepc_dconv import SEPCConv
# from mmdet.ops.dcn.sepc_dconv import ModulatedSEPCConv as SEPCConv


class SEPC(nn.Module):

    def __init__(self,
                 in_channels=[256] * 5,
                 out_channels=256,
                 num_outs=5,
                 stacked_convs=4,
                 pconv_deform=False,
                 lcconv_deform=False,
                 ibn=False,
                 pnorm_cfg=dict(type='BN', requires_grad=True),
                 lcnorm_cfg=dict(type='BN', requires_grad=True),
                 pnorm_eval=True,
                 lcnorm_eval=True,
                 lcconv_padding=0):
        super(SEPC, self).__init__()
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        assert num_outs == 5
        self.fp16_enabled = False
        self.ibn = ibn
        self.pnorm_cfg = pnorm_cfg
        self.lcnorm_cfg = lcnorm_cfg
        self.pnorm_eval = pnorm_eval
        self.lcnorm_eval = lcnorm_eval
        self.pconvs = nn.ModuleList()

        for i in range(stacked_convs):
            self.pconvs.append(
                PConvModule(
                    in_channels[i],
                    out_channels,
                    ibn=self.ibn,
                    norm_cfg=self.pnorm_cfg,
                    norm_eval=self.pnorm_eval,
                    part_deform=pconv_deform))

        self.lconv = SEPCConv(
            256,
            256,
            kernel_size=3,
            padding=lcconv_padding,
            dilation=1,
            part_deform=lcconv_deform)
        # self.cconv = SEPCConv(
        #     256,
        #     256,
        #     kernel_size=3,
        #     padding=lcconv_padding,
        #     dilation=1,
        #     part_deform=lcconv_deform)
        if self.ibn:
            self.lnorm_name, lnorm = build_norm_layer(
                self.lcnorm_cfg, 256, postfix='_loc')
            # self.cnorm_name, cnorm = build_norm_layer(
            #     self.lcnorm_cfg, 256, postfix='_cls')
            self.add_module(self.lnorm_name, lnorm)
            # self.add_module(self.cnorm_name, cnorm)
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        for str in ['l']:
            m = getattr(self, str + 'conv')
            nn.init.normal_(m.weight.data, 0, 0.01)
            if m.bias is not None:
                m.bias.data.zero_()

    @property
    def lnorm(self):
        """nn.Module: normalization layer after localization conv layer"""
        return getattr(self, self.lnorm_name)

    @property
    def cnorm(self):
        """nn.Module: normalization layer after classification conv layer"""
        return getattr(self, self.cnorm_name)

    @auto_fp16()
    def forward(self, inputs):
        assert len(inputs) == len(self.in_channels)
        x = inputs
        for pconv in self.pconvs:
            x = pconv(x)
        # cls_feats = [self.cconv(level, item) for level, item in enumerate(x)]
        loc_feats = [self.lconv(level, item) for level, item in enumerate(x)]
        if self.ibn:
            # cls_feats = integrated_bn(cls_feats, self.cnorm)
            loc_feats = integrated_bn(loc_feats, self.lnorm)
        outs = [self.relu(loc_feat)
                for loc_feat in loc_feats]
        return tuple(outs)

    def train(self, mode=True):
        """Convert the model into training mode while keep normalization layer
        freezed."""
        super(SEPC, self).train(mode)
        if mode and self.lcnorm_eval:
            for m in self.modules():
                # trick: eval have effect on BatchNorm only
                if isinstance(m, _BatchNorm):
                    m.eval()


class PConvModule(nn.Module):

    def __init__(self,
                 in_channels=256,
                 out_channels=256,
                 kernel_size=[3, 3, 3],
                 dilation=[1, 1, 1],
                 groups=[1, 1, 1],
                 ibn=False,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 norm_eval=True,
                 part_deform=False):
        super(PConvModule, self).__init__()

        self.ibn = ibn
        self.norm_cfg = norm_cfg
        self.norm_eval = norm_eval
        self.pconv = nn.ModuleList()
        self.pconv.append(
            SEPCConv(
                in_channels,
                out_channels,
                kernel_size=kernel_size[0],
                dilation=dilation[0],
                groups=groups[0],
                padding=(kernel_size[0] + (dilation[0] - 1) * 2) // 2,
                part_deform=part_deform))
        self.pconv.append(
            SEPCConv(
                in_channels,
                out_channels,
                kernel_size=kernel_size[1],
                dilation=dilation[1],
                groups=groups[1],
                padding=(kernel_size[1] + (dilation[1] - 1) * 2) // 2,
                part_deform=part_deform))
        self.pconv.append(
            SEPCConv(
                in_channels,
                out_channels,
                kernel_size=kernel_size[2],
                dilation=dilation[2],
                groups=groups[2],
                padding=(kernel_size[2] + (dilation[2] - 1) * 2) // 2,
                stride=2,
                part_deform=part_deform))

        if self.ibn:
            self.pnorm_name, pnorm = build_norm_layer(self.norm_cfg, 256)
            self.add_module(self.pnorm_name, pnorm)

        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        for m in self.pconv:
            nn.init.normal_(m.weight.data, 0, 0.01)
            if m.bias is not None:
                m.bias.data.zero_()

    @property
    def pnorm(self):
        """nn.Module: integrated normalization layer after pyramid conv layer
        """
        return getattr(self, self.pnorm_name)

    def forward(self, x):
        next_x = []
        for level, feature in enumerate(x):
            temp_fea = self.pconv[1](level, feature)
            if level > 0:
                temp_fea += self.pconv[2](level, x[level - 1])
            if level < len(x) - 1:
                temp_fea += F.interpolate(
                    self.pconv[0](level, x[level + 1]),
                    size=[temp_fea.size(2), temp_fea.size(3)],
                    mode='bilinear',
                    align_corners=True)
            next_x.append(temp_fea)
        if self.ibn:
            next_x = integrated_bn(next_x, self.pnorm)
        next_x = [self.relu(item) for item in next_x]
        return next_x

    def train(self, mode=True):
        """Convert the model into training mode while keep normalization layer
        freezed."""
        super(PConvModule, self).train(mode)
        if mode and self.norm_eval:
            for m in self.modules():
                # trick: eval have effect on BatchNorm only
                if isinstance(m, _BatchNorm):
                    m.eval()


class SEPC_Decoder(nn.Module):

    def __init__(self,
                 in_channels=[256] * 6,
                 out_channels=256,
                 num_outs=5,
                 stacked_convs=6,
                 pconv_deform=False,
                 lcconv_deform=False,
                 ibn=False,
                 pnorm_cfg=dict(type='BN', requires_grad=True),
                 lcnorm_cfg=dict(type='BN', requires_grad=True),
                 pnorm_eval=True,
                 lcnorm_eval=True,
                 lcconv_padding=0):
        super(SEPC_Decoder, self).__init__()
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        assert num_outs == 5
        self.fp16_enabled = False
        self.ibn = ibn
        self.pnorm_cfg = pnorm_cfg
        self.lcnorm_cfg = lcnorm_cfg
        self.pnorm_eval = pnorm_eval
        self.lcnorm_eval = lcnorm_eval
        self.pconvs = nn.ModuleList()

        for i in range(stacked_convs):
            self.pconvs.append(
                PConvModule(
                    in_channels[i],
                    out_channels,
                    ibn=self.ibn,
                    norm_cfg=self.pnorm_cfg,
                    norm_eval=self.pnorm_eval,
                    part_deform=pconv_deform))

        # self.lconv = SEPCConv(
        #     256,
        #     256,
        #     kernel_size=3,
        #     padding=lcconv_padding,
        #     dilation=1,
        #     part_deform=lcconv_deform)
        # self.cconv = SEPCConv(
        #     256,
        #     256,
        #     kernel_size=3,
        #     padding=lcconv_padding,
        #     dilation=1,
        #     part_deform=lcconv_deform)
        if self.ibn:
            self.lnorm_name, lnorm = build_norm_layer(
                self.lcnorm_cfg, 256, postfix='_loc')
            # self.cnorm_name, cnorm = build_norm_layer(
            #     self.lcnorm_cfg, 256, postfix='_cls')
            self.add_module(self.lnorm_name, lnorm)
            # self.add_module(self.cnorm_name, cnorm)
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        # for str in ['l']:
        #     m = getattr(self, str + 'conv')
        #     nn.init.normal_(m.weight.data, 0, 0.01)
        #     if m.bias is not None:
        #         m.bias.data.zero_()
        pass

    @property
    def lnorm(self):
        """nn.Module: normalization layer after localization conv layer"""
        return getattr(self, self.lnorm_name)

    @property
    def cnorm(self):
        """nn.Module: normalization layer after classification conv layer"""
        return getattr(self, self.cnorm_name)

    @auto_fp16()
    def forward(self, inputs):
        # assert len(inputs) == len(self.in_channels)
        x = inputs
        step=0
        out_f=[]
        for pconv in self.pconvs:
            x = pconv(x)
            if step>=2:
                out_f.append(x[0])
                x=x[1:]
            x=[F.upsample(i, scale_factor=2, mode='bilinear', align_corners=True) for i in x]
            step+=1

        out_f.append(x[0])
        # cls_feats = [self.cconv(level, item) for level, item in enumerate(x)]
        # loc_feats = [self.lconv(level, item) for level, item in enumerate(x)]
        if self.ibn:
            # cls_feats = integrated_bn(cls_feats, self.cnorm)
            # loc_feats = integrated_bn(loc_feats, self.lnorm)
            pass
        outs = [self.relu(feat)
                for feat in out_f]
        return tuple(outs)

    def train(self, mode=True):
        """Convert the model into training mode while keep normalization layer
        freezed."""
        super(SEPC_Decoder, self).train(mode)
        if mode and self.lcnorm_eval:
            for m in self.modules():
                # trick: eval have effect on BatchNorm only
                if isinstance(m, _BatchNorm):
                    m.eval()






def integrated_bn(fms, bn):
    sizes = [p.shape[2:] for p in fms]
    n, c = fms[0].shape[0], fms[0].shape[1]
    fm = torch.cat([p.view(n, c, 1, -1) for p in fms], dim=-1)
    fm = bn(fm)
    fm = torch.split(fm, [s[0] * s[1] for s in sizes], dim=-1)
    return [p.view(n, c, s[0], s[1]) for p, s in zip(fm, sizes)]