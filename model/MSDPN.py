import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers import *


class EBlock(nn.Module):
    def __init__(self, out_channel, num_res=8):
        super(EBlock, self).__init__()

        layers = [ResBlock(out_channel, out_channel) for _ in range(num_res - 1)]
        layers.append(ResBlock(out_channel, out_channel, filter=True))
        # layers.append(WTConv2d(out_channel, out_channel))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class DBlock(nn.Module):
    def __init__(self, channel, num_res=8):
        super(DBlock, self).__init__()

        layers = [ResBlock(channel, channel) for _ in range(num_res - 1)]
        layers.append(ResBlock(channel, channel, filter=True))
        # layers.append(WTConv2d(channel, channel))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)




class SCM(nn.Module):
    def __init__(self, out_plane):
        super(SCM, self).__init__()
        self.main = nn.Sequential(
            BasicConv(3, out_plane // 4, kernel_size=3, stride=1, relu=True),
            BasicConv(out_plane // 4, out_plane // 2, kernel_size=1, stride=1, relu=True),
            BasicConv(out_plane // 2, out_plane // 2, kernel_size=3, stride=1, relu=True),
            BasicConv(out_plane // 2, out_plane, kernel_size=1, stride=1, relu=False),
            nn.InstanceNorm2d(out_plane, affine=True)
        )

    def forward(self, x):
        x = self.main(x)
        return x



class FAM(nn.Module):
    def __init__(self, channel):
        super(FAM, self).__init__()
        self.merge = BasicConv(channel * 2, channel, kernel_size=3, stride=1, relu=False)

    def forward(self, x1, x2):
        return self.merge(torch.cat([x1, x2], dim=1))


class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.sa = nn.Conv2d(2, 1, 7, padding=3, padding_mode='reflect', bias=True)

    def forward(self, x):
        x_avg = torch.mean(x, dim=1, keepdim=True)
        x_max, _ = torch.max(x, dim=1, keepdim=True)
        x2 = torch.cat([x_avg, x_max], dim=1)
        sattn = self.sa(x2)
        return sattn

class AFFM(nn.Module):  # Cross Attention Fusion Module
    def __init__(self, channel):
        super(AFFM, self).__init__()

        # Spatial attention
        self.SA = SpatialAttention()

        # Convolution layers for feature processing
        self.avg1 = nn.Conv2d(channel, 64, 1, stride=1, padding=0)
        self.avg2 = nn.Conv2d(channel, 64, 1, stride=1, padding=0)
        self.max1 = nn.Conv2d(channel, 64, 1, stride=1, padding=0)
        self.max2 = nn.Conv2d(channel, 64, 1, stride=1, padding=0)

        self.avg11 = nn.Conv2d(64, channel, 1, stride=1, padding=0)
        self.avg22 = nn.Conv2d(64, channel, 1, stride=1, padding=0)
        self.max11 = nn.Conv2d(64, channel, 1, stride=1, padding=0)
        self.max22 = nn.Conv2d(64, channel, 1, stride=1, padding=0)

    def _process_feature(self, x, avg_conv, max_conv, avg_up_conv, max_up_conv):
        # Compute average and max features
        avg = torch.mean(x, dim=-1, keepdim=True).unsqueeze(-1)
        max, _ = torch.max(x, dim=-1, keepdim=True)
        max = max.unsqueeze(-1)

        # Process features through conv layers
        avg = F.relu(avg_conv(avg))
        max = F.relu(max_conv(max))
        avg = avg_up_conv(avg).squeeze(-1)
        max = max_up_conv(max).squeeze(-1)

        return avg + max

    def forward(self, f1, f2):
        b, c, h, w = f1.size()

        # Reshape features
        f1 = f1.reshape([b, c, -1])  # [b, c, h*w]
        f2 = f2.reshape([b, c, -1])  # [b, c, h*w]

        # Process features
        a1 = self._process_feature(f1, self.avg1, self.max1, self.avg11, self.max11)
        a2 = self._process_feature(f2, self.avg2, self.max2, self.avg22, self.max22)

        # Cross attention
        cross = torch.matmul(a1, a2.transpose(1, 2))
        a1 = torch.matmul(F.softmax(cross, dim=-1), f1)
        a2 = torch.matmul(F.softmax(cross.transpose(1, 2), dim=-1), f2)

        # Reshape back to original shape
        a1 = a1.reshape([b, c, h, w])
        a2 = a2.reshape([b, c, h, w])

        # Fuse features
        f = a1 + a2

        # Apply spatial attention
        attention_value = self.SA(f)
        out = f * attention_value  # Broadcast attention to all channels
        out = out + f  # Residual connection

        return out






class MSDPN(nn.Module):
    def __init__(self, num_res=4):
        super(MSDPN, self).__init__()

        base_channel = 16

        self.Encoder = nn.ModuleList([
            EBlock(base_channel, num_res),
            EBlock(base_channel * 2, num_res),
            EBlock(base_channel * 4, num_res),
        ])

        self.feat_extract = nn.ModuleList([
            BasicConv(3, base_channel, kernel_size=3, relu=True, stride=1),
            BasicConv(base_channel, base_channel * 2, kernel_size=3, relu=True, stride=2),
            BasicConv(base_channel * 2, base_channel * 4, kernel_size=3, relu=True, stride=2),
            BasicConv(base_channel * 4, base_channel * 2, kernel_size=4, relu=True, stride=2, transpose=True),
            BasicConv(base_channel * 2, base_channel, kernel_size=4, relu=True, stride=2, transpose=True),
            BasicConv(base_channel, 3, kernel_size=3, relu=False, stride=1)
        ])

        self.Decoder = nn.ModuleList([
            DBlock(base_channel * 4, num_res),
            DBlock(base_channel * 2, num_res),
            DBlock(base_channel, num_res)
        ])

        self.Convs = nn.ModuleList([
            BasicConv(base_channel * 4, base_channel * 2, kernel_size=1, relu=True, stride=1),
            BasicConv(base_channel * 2, base_channel, kernel_size=1, relu=True, stride=1),
        ])

        self.ConvsOut = nn.ModuleList(
            [
                BasicConv(base_channel * 4, 3, kernel_size=3, relu=False, stride=1),
                BasicConv(base_channel * 2, 3, kernel_size=3, relu=False, stride=1),
            ]
        )

        self.FAM1 = FAM(base_channel * 4)
        self.SCM1 = SCM(base_channel * 4)
        self.FAM2 = FAM(base_channel * 2)
        self.SCM2 = SCM(base_channel * 2)

        self.fusion1=AFFM(base_channel)
        self.fusion2 =AFFM(base_channel * 2)
    def forward(self, x):
        x_2 = F.interpolate(x, scale_factor=0.5)
        x_4 = F.interpolate(x_2, scale_factor=0.5)
        z2 = self.SCM2(x_2)
        z4 = self.SCM1(x_4)

        outputs = list()
        # 256
        x_ = self.feat_extract[0](x)
        res1 = self.Encoder[0](x_)
        # 128
        z = self.feat_extract[1](res1)
        z = self.FAM2(z, z2)
        res2 = self.Encoder[1](z)
        # 64
        z = self.feat_extract[2](res2)
        z = self.FAM1(z, z4)
        z = self.Encoder[2](z)

        z = self.Decoder[0](z)
        z_ = self.ConvsOut[0](z)
        # 128
        z = self.feat_extract[3](z)
        outputs.append(z_ + x_4)
        z = self.fusion2(z, res2)
        # z = torch.cat([z, res2], dim=1)
        # z=self.fusion(z,res2)
        # z=self.att1(z)
        # z = self.Convs[0](z)

        z = self.Decoder[1](z)
        z_ = self.ConvsOut[1](z)
        # 256
        z = self.feat_extract[4](z)
        outputs.append(z_ + x_2)
        # z=self.fusion(z,res1)
        # z = torch.cat([z, res1], dim=1)
        # # z = self.att2(z)
        # z = self.Convs[1](z)
        z = self.fusion1(z, res1)
        z = self.Decoder[2](z)
        z = self.feat_extract[5](z)
        outputs.append(z + x)

        return outputs


def build_net():
    return MSDPN()
