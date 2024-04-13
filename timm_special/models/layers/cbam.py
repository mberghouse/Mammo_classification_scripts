# """ CBAM (sort-of) Attention

# Experimental impl of CBAM: Convolutional Block Attention Module: https://arxiv.org/abs/1807.06521

# WARNING: Results with these attention layers have been mixed. They can significantly reduce performance on
# some tasks, especially fine-grained it seems. I may end up removing this impl.

# Hacked together by / Copyright 2020 Ross Wightman
# """
import torch
from torch import nn as nn
import torch.nn.functional as F

from .conv_bn_act import ConvNormAct
from .create_act import create_act_layer, get_act_layer
from .helpers import make_divisible


class ChannelAttn(nn.Module):
    """ Original CBAM channel attention module, currently avg + max pool variant only.
    """
    def __init__(
            self, channels, rd_ratio=.25, rd_channels=None, rd_divisor=1,
            act_layer=nn.ReLU, gate_layer='sigmoid', mlp_bias=False):
        super(ChannelAttn, self).__init__()
        if not rd_channels:
            rd_channels = make_divisible(channels * rd_ratio, rd_divisor, round_limit=0.)
        self.fc1 = nn.Conv2d(channels, rd_channels, 1, bias=mlp_bias)
        self.act = nn.LeakyReLU(.1, inplace=True)
        self.fc2 = nn.Conv2d(rd_channels, channels, 1, bias=mlp_bias)
        self.gate = create_act_layer(gate_layer)

    def forward(self, x):
        x_avg = self.fc2(self.act(self.fc1(x.mean((2, 3), keepdim=True))))
        x_max = self.fc2(self.act(self.fc1(x.amax((2, 3), keepdim=True))))
        return x * self.gate(x_avg + x_max)


class LightChannelAttn(ChannelAttn):
    """An experimental 'lightweight' that sums avg + max pool first
    """
    def __init__(
            self, channels, rd_ratio=1./4, rd_channels=None, rd_divisor=1,
            act_layer=nn.ReLU, gate_layer='sigmoid', mlp_bias=False):
        super(LightChannelAttn, self).__init__(
            channels, rd_ratio, rd_channels, rd_divisor, act_layer, gate_layer, mlp_bias)

    def forward(self, x):
        x_pool = 0.5 * x.mean((2, 3), keepdim=True) + 0.5 * x.amax((2, 3), keepdim=True)
        x_attn = self.fc2(self.act(self.fc1(x_pool)))
        return x * F.sigmoid(x_attn)


class SpatialAttn(nn.Module):
    """ Original CBAM spatial attention module
    """
    def __init__(self, kernel_size=3, gate_layer='sigmoid'):
        super(SpatialAttn, self).__init__()
        self.conv = ConvNormAct(2, 1, kernel_size, padding = (kernel_size-1)//2, apply_act=False)
        self.gate = create_act_layer(gate_layer)

    def forward(self, x):
        x_attn = torch.cat([x.mean(dim=1, keepdim=True), x.amax(dim=1, keepdim=True)], dim=1)
        x_attn = self.conv(x_attn)
        return x * self.gate(x_attn)


class LightSpatialAttn(nn.Module):
    """An experimental 'lightweight' variant that sums avg_pool and max_pool results.
    """
    def __init__(self, kernel_size=7, gate_layer='sigmoid'):
        super(LightSpatialAttn, self).__init__()
        self.conv = ConvNormAct(1, 1, kernel_size, apply_act=False)
        self.gate = create_act_layer(gate_layer)

    def forward(self, x):
        x_attn = 0.5 * x.mean(dim=1, keepdim=True) + 0.5 * x.amax(dim=1, keepdim=True)
        x_attn = self.conv(x_attn)
        return x * self.gate(x_attn)


# class CbamModule(nn.Module):
    # def __init__(
            # self, channels, rd_ratio=1, rd_channels=None, rd_divisor=1,
            # spatial_kernel_size=7, act_layer=nn.ReLU, gate_layer='sigmoid', mlp_bias=False):
        # super(CbamModule, self).__init__()
        # self.channel = ChannelAttn(
            # channels, rd_ratio=rd_ratio, rd_channels=rd_channels,
            # rd_divisor=rd_divisor, act_layer=act_layer, gate_layer=gate_layer, mlp_bias=mlp_bias)
        # self.spatial = SpatialAttn(spatial_kernel_size, gate_layer=gate_layer)

    # def forward(self, x):
        # x = self.channel(x)
        # x = self.spatial(x)
        # return x


class LightCbamModule(nn.Module):
    def __init__(
            self, channels, rd_ratio=1./16, rd_channels=None, rd_divisor=1,
            spatial_kernel_size=7, act_layer=nn.ReLU, gate_layer='sigmoid', mlp_bias=False):
        super(LightCbamModule, self).__init__()
        self.channel = LightChannelAttn(
            channels, rd_ratio=rd_ratio, rd_channels=rd_channels,
            rd_divisor=rd_divisor, act_layer=act_layer, gate_layer=gate_layer, mlp_bias=mlp_bias)
        self.spatial = LightSpatialAttn(spatial_kernel_size)

    def forward(self, x):
        x = self.channel(x)
        x = self.spatial(x)
        return x

import torch
import math
import torch.nn as nn
import torch.nn.functional as F

class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes,eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.LeakyReLU(.1) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class ChannelGate(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=4, pool_types=['avg', 'max']):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.LeakyReLU(.1),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)
            )
        self.pool_types = pool_types
    def forward(self, x):
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type=='avg':
                avg_pool = F.avg_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( avg_pool )
            elif pool_type=='max':
                max_pool = F.max_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( max_pool )
            elif pool_type=='lp':
                lp_pool = F.lp_pool2d( x, 2, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( lp_pool )
            elif pool_type=='lse':
                # LSE pool only
                lse_pool = logsumexp_2d(x)
                channel_att_raw = self.mlp( lse_pool )

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw

        scale = F.sigmoid( channel_att_sum ).unsqueeze(2).unsqueeze(3).expand_as(x)
        return x * scale

def logsumexp_2d(tensor):
    tensor_flatten = tensor.view(tensor.size(0), tensor.size(1), -1)
    s, _ = torch.max(tensor_flatten, dim=2, keepdim=True)
    outputs = s + (tensor_flatten - s).exp().sum(dim=2, keepdim=True).log()
    return outputs

class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat( (torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1 )

class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size-1) // 2, relu=False)
    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = F.sigmoid(x_out) # broadcasting
        return x * scale

class CbamModule(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=4, pool_types=['avg', 'max'], no_spatial=False):
        super(CbamModule, self).__init__()
        self.ChannelGate = ChannelGate(gate_channels, reduction_ratio, pool_types)
        self.no_spatial=no_spatial
        if not no_spatial:
            self.SpatialGate = SpatialGate()
    def forward(self, x):
        x_out = self.ChannelGate(x)
        if not self.no_spatial:
            x_out = self.SpatialGate(x_out)
        return x_out
