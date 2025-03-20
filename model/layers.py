import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from matplotlib import pyplot as plt
from torch import Tensor

from Dehazing.ITS.feature import SaveFImg




class BasicConv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, bias=True, norm=False, relu=True, transpose=False):
        super(BasicConv, self).__init__()
        if bias and norm:
            bias = False

        padding = kernel_size // 2
        layers = list()
        if transpose:
            padding = kernel_size // 2 - 1
            layers.append(
                nn.ConvTranspose2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias))
        else:
            layers.append(
                nn.Conv2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias))
        if norm:
            layers.append(nn.BatchNorm2d(out_channel))
        if relu:
            layers.append(nn.GELU())
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)


class Gap(nn.Module):
    def __init__(self, in_channel) -> None:
        super().__init__()

        self.fscale_d = nn.Parameter(torch.zeros(in_channel), requires_grad=True)
        self.fscale_h = nn.Parameter(torch.zeros(in_channel), requires_grad=True)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x_d = self.gap(x)
        x_h = (x - x_d) * (self.fscale_h[None, :, None, None] + 1.)
        x_d = x_d * self.fscale_d[None, :, None, None]
        return x_d + x_h



class GCAM(nn.Module):
    def __init__(self, dim):
        super().__init__()

        # 确保 dim 是偶数
        assert dim % 2 == 0, "dim 必须是偶数"

        # 分支1: 空洞卷积和多尺度特征融合
        self.norm1 = nn.BatchNorm2d(dim // 2)
        self.conv1_1 = nn.Conv2d(dim // 2, dim // 2, kernel_size=1)  # 1x1 卷积
        self.conv1_2 = nn.Conv2d(dim // 2, dim // 2, kernel_size=5, padding=2, padding_mode='reflect')  # 5x5 卷积

        # 空洞卷积，扩张率分别为 1, 2, 3
        self.conv3_1 = nn.Conv2d(dim // 2, dim // 2, kernel_size=3, padding=1, dilation=1, padding_mode='reflect')  # 扩张率 = 1
        self.conv3_2 = nn.Conv2d(dim // 2, dim // 2, kernel_size=3, padding=2, dilation=2, padding_mode='reflect')  # 扩张率 = 2
        self.conv3_3 = nn.Conv2d(dim // 2, dim // 2, kernel_size=3, padding=3, dilation=3, padding_mode='reflect')  # 扩张率 = 3

        # 多尺度特征融合
        self.mlp = nn.Sequential(
            nn.Conv2d((dim // 2) * 3, dim * 4, 1),  # 输入通道数为 (dim//2)*3
            nn.GELU(),
            nn.Conv2d(dim * 4, dim // 2, 1)  # 输出通道数为 dim//2
        )

        # 分支2: 注意力机制
        self.pool = nn.AvgPool2d(kernel_size=4, stride=4)  # 4倍下采样
        self.conv2_1 = nn.Conv2d(dim // 2, dim // 2, kernel_size=3, padding=1)  # 3x3 卷积
        self.upsample = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)  # 4倍上采样
        self.sigmoid = nn.Sigmoid()  # 归一化

        # 分支2: 直接进行 3x3 卷积
        self.conv2_2 = nn.Conv2d(dim // 2, dim // 2, kernel_size=3, padding=1)

        # 最终 3x3 卷积
        self.conv3_final = nn.Conv2d(dim // 2, dim // 2, kernel_size=3, padding=1)

    def forward(self, x):
        identity = x
        part1, part2 = torch.chunk(x, chunks=2, dim=1)  # 将输入特征图分为两部分
        identity1 = part1

        # 分支1: 空洞卷积和多尺度特征融合
        part1 = self.norm1(part1)
        part1 = self.conv1_1(part1)
        part1 = self.conv1_2(part1)
        part1 = torch.cat([self.conv3_1(part1), self.conv3_2(part1), self.conv3_3(part1)], dim=1)  # 多尺度特征融合
        part1 = self.mlp(part1)  # 特征变换
        part1 = identity1 + part1  # 残差连接

        # 分支2: 注意力机制
        attn = self.pool(part2)  # 4倍下采样
        attn = self.conv2_1(attn)  # 3x3 卷积
        attn = self.upsample(attn)  # 4倍上采样
        attn = self.sigmoid(attn)  # 归一化

        # 计算主分支
        feat = self.conv2_2(part2)  # 3x3 卷积
        enhanced = feat * attn  # 逐元素乘法
        res = self.conv3_final(enhanced)  # 最终卷积
        part2 = part2 + res  # 残差连接

        # 合并分支1和分支2
        x = torch.cat((part1, part2), dim=1)

        return x


class ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel, filter=False):
        super(ResBlock, self).__init__()
        self.conv1 = BasicConv(in_channel, out_channel, kernel_size=3, stride=1, relu=True)
        self.conv2 = BasicConv(out_channel, out_channel, kernel_size=3, stride=1, relu=False)
        self.filter = filter

        self.dyna = dynamic_filter(in_channel // 2) if filter else nn.Identity()
        self.dyna_2 = dynamic_filter(in_channel // 2, kernel_size=5) if filter else nn.Identity()
        self.gcam = GCAM(out_channel)
        # self.WtConv2d=WTConv2d(out_channel, out_channel)
        # self.window=AvgPoolUpsampleBlock(out_channel)
        self.dim_conv = out_channel

    def forward(self, x):
        out = self.conv1(x)

        if self.filter:
            k3, k5 = torch.chunk(out, 2, dim=1)
            out_k3 = self.dyna(k3)
            out_k5 = self.dyna_2(k5)
            out = torch.cat((out_k3, out_k5), dim=1)

        out = self.gcam(out)
        out = self.conv2(out)
        return out + x

class dynamic_filter(nn.Module):
    def __init__(self, inchannels, kernel_size=3, stride=1, group=8):
        super(dynamic_filter, self).__init__()
        self.stride = stride
        self.kernel_size = kernel_size
        self.group = group

        self.conv = nn.Conv2d(inchannels, group*kernel_size**2, kernel_size=1, stride=1, bias=False)
        self.bn = nn.BatchNorm2d(group*kernel_size**2)
        self.act = nn.Softmax(dim=-2)
        nn.init.kaiming_normal_(self.conv.weight, mode='fan_out', nonlinearity='relu')
        self.lamb_l = nn.Parameter(torch.zeros(inchannels), requires_grad=True)
        self.lamb_h = nn.Parameter(torch.zeros(inchannels), requires_grad=True)
        self.pad = nn.ReflectionPad2d(kernel_size//2)

        self.ap = nn.AdaptiveAvgPool2d((1, 1))
        self.modulate = SFconv(inchannels)


    def forward(self, x):
        identity_input = x # 3,32,64,64
        low_filter = self.ap(x)
        low_filter = self.conv(low_filter)
        low_filter = self.bn(low_filter)

        n, c, h, w = x.shape  
        x = F.unfold(self.pad(x), kernel_size=self.kernel_size).reshape(n, self.group, c//self.group, self.kernel_size**2, h*w)

        n,c1,p,q = low_filter.shape
        low_filter = low_filter.reshape(n, c1//self.kernel_size**2, self.kernel_size**2, p*q).unsqueeze(2)
       
        low_filter = self.act(low_filter)
    
        low_part = torch.sum(x * low_filter, dim=3).reshape(n, c, h, w)

        out_high = identity_input - low_part
        # plt.subplot(1, 1, 1)
        # p = out_high  # b,c,h,w
        # data = p[0].squeeze(0).permute(1, 2, 0).cpu().detach().numpy()  # h,w,c
        # data = np.mean(data, axis=2)  # h,w,1
        # # # 创建图形和3D轴
        # plt.imshow(data, cmap='jet', interpolation='nearest')
        #
        # plt.subplot(1, 3, 2)
        # p = low_part # b,c,h,w
        # data = p[0].squeeze(0).permute(1, 2, 0).cpu().detach().numpy()  # h,w,c
        # data = np.mean(data, axis=2)  # h,w,1
        # # # # 创建图形和3D轴
        # plt.imshow(data, cmap='jet', interpolation='nearest')

        out = self.modulate(low_part, out_high)
        # SaveFImg([out_high],['高频'])
        # plt.subplot(1, 3, 3)
        # p = out  # b,c,h,w
        # data = p[0].squeeze(0).permute(1, 2, 0).cpu().detach().numpy()  # h,w,c
        # data = np.mean(data, axis=2)  # h,w,1
        # # # # 创建图形和3D轴
        # plt.imshow(data, cmap='jet', interpolation='nearest')
        # plt.show()
        return out


class SFconv(nn.Module):
    def __init__(self, features, M=2, r=2, L=32) -> None:
        super().__init__()

        d = max(int(features/r), L)
        self.features = features

        self.fc = nn.Conv2d(features, d, 1, 1, 0)
        self.fcs = nn.ModuleList([])
        for i in range(M):
            self.fcs.append(
                nn.Conv2d(d, features, 1, 1, 0)
            )
        self.softmax = nn.Softmax(dim=1)
        self.gap = nn.AdaptiveAvgPool2d(1)

        self.out = nn.Conv2d(features, features, 1, 1, 0)
    def forward(self, low, high):
        emerge = low + high
        emerge = self.gap(emerge)

        fea_z = self.fc(emerge)

        high_att = self.fcs[0](fea_z)
        low_att = self.fcs[1](fea_z)

        attention_vectors = torch.cat([high_att, low_att], dim=1)

        attention_vectors = self.softmax(attention_vectors)
        high_att, low_att = torch.chunk(attention_vectors, 2, dim=1)

        fea_high = high * high_att
        fea_low = low * low_att

        out = self.out(fea_high + fea_low)
        return out


