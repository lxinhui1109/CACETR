# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Backbone modules.
"""
from collections import OrderedDict

import torch
import torch.nn.functional as F
import torchvision
from torch import nn
from torchvision.models._utils import IntermediateLayerGetter
from typing import Dict, List

from util.back import NestedTensor, is_main_process

from .position_encoding import build_position_encoding


class FrozenBatchNorm2d(torch.nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.

    Copy-paste from torchvision.misc.ops with added eps before rqsrt,
    without which any other models than torchvision.models.resnet[18,34,50,101]
    produce nans.
    """

    def __init__(self, n):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        num_batches_tracked_key = prefix + 'num_batches_tracked'
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

    def forward(self, x):
        # move reshapes to the beginning
        # to make it fuser-friendly
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = 1e-5
        scale = w * (rv + eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias


class BackboneBase(nn.Module):

    def __init__(self, backbone: nn.Module, train_backbone: bool, num_channels: int, return_interm_layers: bool):
        super().__init__()
        for name, parameter in backbone.named_parameters():
            if not train_backbone or 'layer2' not in name and 'layer3' not in name and 'layer4' not in name:
                parameter.requires_grad_(False)
        if return_interm_layers:
            return_layers = {"layer1": "0", "layer2": "1", "layer3": "2", "layer4": "3"}
        else:
            return_layers = {'layer4': "0"}
        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)
        self.num_channels = num_channels

    def forward(self, tensor_list: NestedTensor):
        # 通过self.body即resnet50获取最后一层卷积得到的张量[2,3,768,768]->[2,2048,24,24]
        if isinstance(tensor_list, NestedTensor):#如果tensor_list是NestedTensor形式，也就是样本
            xs = self.body(tensor_list.tensors)
            out: Dict[str, NestedTensor] = {}
            for name, x in xs.items():
                # 获取原始输入的mask[2,768,768]
                m = tensor_list.mask
                assert m is not None
                # 根据resnet50输出的wh维度进行reshape即[2,768,768]->[2,24,24]
                mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
                out[name] = NestedTensor(x, mask)
            # 此时的输出为{mask,[2,24,24],tensor_list,[2,2048,24,24]}
        else:#tensor_list不是NestedTensor形式 也就是对样例操作
            out = self.body(tensor_list)
        return out


class Backbone(BackboneBase):
    """ResNet backbone with frozen BatchNorm."""

    def __init__(self, name: str,
                 train_backbone: bool,
                 return_layer: str,
                 frozen_bn: bool,
                 dilation: bool):

        if frozen_bn:
            backbone = getattr(torchvision.models, name)(
                replace_stride_with_dilation=[False, False, dilation],
                pretrained=True, norm_layer=FrozenBatchNorm2d)
        else:
            backbone = getattr(torchvision.models, name)(
                replace_stride_with_dilation=[False, False, dilation],
                pretrained=True)

        # load the SwAV pre-training model from the url instead of supervised pre-training model
        if name == 'resnet50':
            checkpoint = torch.hub.load_state_dict_from_url(
                'https://dl.fbaipublicfiles.com/deepcluster/swav_800ep_pretrain.pth.tar', map_location="cpu")
            state_dict = {k.replace("module.", ""): v for k, v in checkpoint.items()}
            backbone.load_state_dict(state_dict, strict=False)
            # pass
        if name in ('resnet18', 'resnet34'):
            num_channels = 512
        else:
            if return_layer == 'layer3':
                num_channels = 1024
            else:
                num_channels = 2048
        super().__init__(backbone, train_backbone, num_channels, return_layer)

#拼接
class Joiner(nn.Sequential):
    def __init__(self, backbone, position_embedding):
        super().__init__(backbone, position_embedding)

    def forward(self, tensor_list):
        """supports both NestedTensor and torch.Tensor
        """
        if isinstance(tensor_list, NestedTensor):  # 输入图像大小[2,3,810,769]
            # xs是Backbone中Resnet50的输出结果
            xs = self[0](tensor_list)
            out: List[NestedTensor] = []
            pos = []
            for name, x in xs.items():
                out.append(x)
                # position encoding 位置编码
                pos.append(self[1](x).to(x.tensors.dtype))
            return out, pos
        else:
            return list(self[0](tensor_list).values())


def build_backbone(cfg):
    position_embedding = build_position_encoding(cfg)
    train_backbone = cfg.TRAIN.lr_backbone > 0
    return_interm_layers = 'layer4'
    backbone = Backbone(cfg.MODEL.backbone, train_backbone, return_interm_layers, cfg.MODEL.fix_bn, cfg.MODEL.dilation)
    model = Joiner(backbone, position_embedding)
    model.num_channels = backbone.num_channels
    return model


if __name__ == '__main__':
    backbone = Backbone('resnet50',
                        train_backbone=True,
                        return_layer='layer4',
                        frozen_bn=False,
                        dilation=False)

    inputs = torch.rand(5, 3, 256, 256)
    outputs = backbone(inputs)
