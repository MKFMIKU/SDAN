import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import DeformConv2d

from models.networks import PatchAttn


def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size // 2), bias=bias)


class MeanShift(nn.Conv2d):
    def __init__(
            self, rgb_range,
            rgb_mean=(0.4488, 0.4371, 0.4040), rgb_std=(1.0, 1.0, 1.0), sign=-1):
        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1) / std.view(3, 1, 1, 1)
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean) / std
        for p in self.parameters():
            p.requires_grad = False


class BasicBlock(nn.Sequential):
    def __init__(
            self, conv, in_channels, out_channels, kernel_size, stride=1, bias=False,
            bn=True, act=nn.ReLU(True)):

        m = [conv(in_channels, out_channels, kernel_size, bias=bias)]
        if bn:
            m.append(nn.BatchNorm2d(out_channels))
        if act is not None:
            m.append(act)

        super(BasicBlock, self).__init__(*m)


class ResBlock(nn.Module):
    def __init__(
            self, conv, n_feats, kernel_size,
            bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(ResBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(conv(n_feats, n_feats, kernel_size, bias=bias))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if i == 0:
                m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x
        return res


class Sq_DFS_Attention(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int,
            groups: int = 1,
            bias: bool = True
    ):

        super(Sq_DFS_Attention, self).__init__()

        self.kernel_size = kernel_size
        self.groups = groups

        self.ref_conv = nn.Sequential(
            nn.Conv2d(3 * 4, in_channels, 3, stride=1, padding=1, bias=bias),
            nn.ReLU(inplace=True)
        )

        self.offset_conv = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, 3, stride=1, padding=1, bias=bias),
            PatchAttn(in_channels, 8),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, 3, stride=1, padding=1, bias=bias),
            PatchAttn(in_channels, 8),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, groups * 2, 3, stride=1, padding=1, bias=bias),
        )
        self.deform_conv = DeformConv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=kernel_size // 2,
            padding=kernel_size % 2,
            groups=groups,
            bias=bias)
        self.relu = nn.ReLU(inplace=True)

        self.rgb_conv = nn.Conv2d(in_channels, 3, 3, stride=1, padding=1, bias=bias)

        self.offset_conv.apply(self.init_0)

    def init_0(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, 0.02)
            nn.init.constant_(m.bias, 0)

    def forward(self, x, ref_x=None):
        n, c, h, w = x.size()

        ref_x = self.ref_conv(torch.cat([ref_x,
                                         torch.flip(ref_x, [2]),
                                         torch.flip(ref_x, [3]),
                                         torch.flip(ref_x, [2, 3])], dim=1))
        offset = self.offset_conv(torch.cat([x, ref_x], dim=1))

        with torch.no_grad():
            point_x = torch.linspace(0, h - 1, h).reshape(1, 1, h, 1).repeat(n, 1, 1, w)
            point_y = torch.linspace(0, w - 1, w).reshape(1, 1, 1, w).repeat(n, 1, h, 1)
            point = torch.cat((point_x, point_y), dim=1).type_as(x)
            point = point + offset
            tmp_x = point[:, 0:1, :, :]
            tmp_y = point[:, 1:2, :, :]
            mask = (tmp_x >= 0) * (tmp_x < h) * (tmp_y >= 0) * (tmp_y < w)
            mask = mask.float()

        conv_offset = offset.repeat(1, self.kernel_size * self.kernel_size, 1, 1)
        x = self.relu(self.deform_conv(x, conv_offset))
        rgb = self.rgb_conv(x)
        return x, rgb, mask, offset


class DFS_Attention(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int,
            groups: int = 1,
            bias: bool = True
    ):

        super(DFS_Attention, self).__init__()

        self.kernel_size = kernel_size
        self.groups = groups

        self.ref_conv = nn.Sequential(
            nn.Conv2d(3, in_channels, 3, stride=1, padding=1, bias=bias),
            nn.ReLU(inplace=True)
        )

        self.offset_conv = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, 3, stride=1, padding=1, bias=bias),
            PatchAttn(in_channels, 8),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, 3, stride=1, padding=1, bias=bias),
            PatchAttn(in_channels, 8),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, groups * 2, 3, stride=1, padding=1, bias=bias),
        )
        self.deform_conv = DeformConv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=kernel_size // 2,
            padding=kernel_size % 2,
            groups=groups,
            bias=bias)
        self.relu = nn.ReLU(inplace=True)

        self.rgb_conv = nn.Conv2d(in_channels, 3, 3, stride=1, padding=1, bias=bias)

        self.offset_conv.apply(self.init_0)

    def init_0(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, 0.02)
            nn.init.constant_(m.bias, 0)

    def forward(self, x, ref_x=None):
        n, c, h, w = x.size()

        # ref_x = self.ref_conv(torch.cat([ref_x,
        #                                  torch.flip(ref_x, [2]),
        #                                  torch.flip(ref_x, [3]),
        #                                  torch.flip(ref_x, [2, 3])], dim=1))
        ref_x = self.ref_conv(ref_x)
        offset = self.offset_conv(torch.cat([x, ref_x], dim=1))

        with torch.no_grad():
            point_x = torch.linspace(0, h - 1, h).reshape(1, 1, h, 1).repeat(n, 1, 1, w)
            point_y = torch.linspace(0, w - 1, w).reshape(1, 1, 1, w).repeat(n, 1, h, 1)
            point = torch.cat((point_x, point_y), dim=1).type_as(x)
            point = point + offset
            tmp_x = point[:, 0:1, :, :]
            tmp_y = point[:, 1:2, :, :]
            mask = (tmp_x >= 0) * (tmp_x < h) * (tmp_y >= 0) * (tmp_y < w)
            mask = mask.float()

        conv_offset = offset.repeat(1, self.kernel_size * self.kernel_size, 1, 1)
        x = self.relu(self.deform_conv(x, conv_offset))
        rgb = self.rgb_conv(x)
        return x, rgb, mask, offset


class ChannelAttention(nn.Module):
    """Channel attention used in RCAN.
    Args:
        num_feat (int): Channel number of intermediate features.
        squeeze_factor (int): Channel squeeze factor. Default: 16.
    """

    def __init__(self, num_feat, squeeze_factor=16):
        super(ChannelAttention, self).__init__()
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(num_feat, num_feat // squeeze_factor, 1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_feat // squeeze_factor, num_feat, 1, padding=0),
            nn.Sigmoid())

    def forward(self, x):
        y = self.attention(x)
        return x * y


class DFS_CL_Attention(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int,
            groups: int = 1,
            bias: bool = True
    ):

        super(DFS_CL_Attention, self).__init__()

        self.kernel_size = kernel_size
        self.groups = groups

        self.ref_conv = nn.Sequential(
            nn.Conv2d(3 * 4, in_channels, 3, stride=1, padding=1, bias=bias),
            nn.ReLU(inplace=True)
        )

        self.offset_conv = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, 3, stride=1, padding=1, bias=bias),
            nn.ReLU(inplace=True),
            ChannelAttention(in_channels),
            nn.Conv2d(in_channels, in_channels, 3, stride=1, padding=1, bias=bias),
            nn.ReLU(inplace=True),
            ChannelAttention(in_channels),
            nn.Conv2d(in_channels, groups * 2 * kernel_size * kernel_size, 3, stride=1, padding=1, bias=bias),
        )
        self.deform_conv = DeformConv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=kernel_size // 2,
            padding=kernel_size % 2,
            groups=groups,
            bias=bias)
        self.relu = nn.ReLU(inplace=True)

        self.rgb_conv = nn.Conv2d(in_channels, 3, 3, stride=1, padding=1, bias=bias)

        self.offset_conv.apply(self.init_0)

    def init_0(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, 0.02)
            nn.init.constant_(m.bias, 0)

    def forward(self, x, ref_x=None):
        n, c, h, w = x.size()

        ref_x = self.ref_conv(torch.cat([ref_x,
                                         torch.flip(ref_x, [2]),
                                         torch.flip(ref_x, [3]),
                                         torch.flip(ref_x, [2, 3])], dim=1))
        offset = self.offset_conv(torch.cat([x, ref_x], dim=1))

        # with torch.no_grad():
        #     point_x = torch.linspace(0, h - 1, h).reshape(1, 1, h, 1).repeat(n, 1, 1, w)
        #     point_y = torch.linspace(0, w - 1, w).reshape(1, 1, 1, w).repeat(n, 1, h, 1)
        #     point = torch.cat((point_x, point_y), dim=1).type_as(x)
        #     point = point + offset
        #     tmp_x = point[:, 0:1, :, :]
        #     tmp_y = point[:, 1:2, :, :]
        #     mask = (tmp_x >= 0) * (tmp_x < h) * (tmp_y >= 0) * (tmp_y < w)
        #     mask = mask.float()
        mask = offset

        # conv_offset = offset.repeat(1, self.kernel_size * self.kernel_size, 1, 1)
        x = self.relu(self.deform_conv(x, offset))
        rgb = self.rgb_conv(x)
        return x, rgb, mask, offset


class DFS(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int,
            groups: int = 1,
            bias: bool = True
    ):

        super(DFS, self).__init__()

        self.kernel_size = kernel_size

        self.offset_conv = nn.Conv2d(in_channels + 3, 2 * kernel_size * kernel_size, 3, stride=1, padding=1, bias=bias)
        self.deform_conv = DeformConv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=kernel_size % 2,
            padding=kernel_size // 2,
            groups=groups,
            bias=bias)

        nn.init.constant_(self.offset_conv.weight, 0)
        nn.init.constant_(self.offset_conv.bias, 0)

    def forward(self, x, ref_x=None):
        n, c, h, w = x.size()
        if self.training:
            offset = self.offset_conv(torch.cat([x, ref_x], dim=1))
            # print("offset", offset.max(), offset.min())
        else:
            offset = torch.zeros((n, 2 * self.kernel_size * self.kernel_size, h, w), device=x.device)
        x = self.deform_conv(x, offset)
        return x


class ResINBlock(nn.Module):
    def __init__(
            self, conv, n_feats, kernel_size, bias=True, nm=False, res_scale=1):

        super(ResINBlock, self).__init__()
        self.conv1 = conv(n_feats, n_feats, kernel_size, bias=bias)
        self.nm1 = nn.BatchNorm2d(n_feats)
        self.act = nn.LeakyReLU(0.2, inplace=True)
        self.conv2 = conv(n_feats, n_feats, kernel_size, bias=bias)
        self.nm2 = nn.BatchNorm2d(n_feats)

        self.res_scale = res_scale

    def forward(self, x):
        ref_x = None
        n, c, h, w = x.size()
        if self.training:
            x, ref_x = torch.split(x, c - 3, dim=1)
        res = self.act(self.nm1(self.conv1(x, ref_x)))
        res = self.nm2(self.conv2(res, ref_x))

        res += x
        if self.training:
            return torch.cat([res, ref_x], dim=1)
        else:
            return res


class ClassicalResINBlock(nn.Module):
    def __init__(
            self, conv, n_feats, kernel_size, bias=False, nm='group', res_scale=1):

        super(ClassicalResINBlock, self).__init__()
        self.conv1 = conv(n_feats, n_feats, kernel_size, bias=bias)
        self.act = nn.ReLU(inplace=True)
        self.conv2 = conv(n_feats, n_feats, kernel_size, bias=bias)

        self.res_scale = res_scale

    def forward(self, x):
        res = self.act(self.conv1(x))
        res = self.conv2(res)

        res += x
        return res


class Upsampler(nn.Sequential):
    def __init__(self, conv, scale, n_feats, bn=False, act=False, bias=True, ps=True):

        m = []
        if (scale & (scale - 1)) == 0:  # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(conv(n_feats, 4 * n_feats, 3, bias))
                m.append(nn.PixelShuffle(2) if ps else nn.ConvTranspose2d(4 * n_feats, n_feats, kernel_size=4, stride=2, padding=1))
                if bn:
                    m.append(nn.BatchNorm2d(n_feats))
                if act == 'relu':
                    m.append(nn.ReLU(True))
                elif act == 'prelu':
                    m.append(nn.PReLU(n_feats))
                elif act == 'lrelu':
                    m.append(nn.LeakyReLU(0.2, inplace=True))

        elif scale == 3:
            m.append(conv(n_feats, 9 * n_feats, 3, bias))
            m.append(nn.PixelShuffle(3))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if act == 'relu':
                m.append(nn.ReLU(True))
            elif act == 'prelu':
                m.append(nn.PReLU(n_feats))
            elif act == 'lrelu':
                m.append(nn.LeakyReLU(0.2, inplace=True))
        else:
            raise NotImplementedError

        super(Upsampler, self).__init__(*m)