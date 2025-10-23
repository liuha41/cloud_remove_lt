from torch import nn as nn
from torch.nn import functional as F

from basicsr.archs.arch_util import default_init_weights
from basicsr.utils.registry import ARCH_REGISTRY


@ARCH_REGISTRY.register()
class CloudRemovalArch(nn.Module):
    """云雾去除架构

    Args:
        num_in_ch (int): 输入通道数. Default: 3.
        num_out_ch (int): 输出通道数. Default: 3.
        num_feat (int): 中间特征通道数. Default: 64.
        num_blocks (int): 残差块数量. Default: 8.
    """

    def __init__(self, num_in_ch=3, num_out_ch=3, num_feat=64, upscale=4):
        super(CloudRemovalArch, self).__init__()
        num_blocks = 8
        self.num_feat = num_feat

        # 浅层特征提取
        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)

        # 深层特征提取 - 使用残差块
        self.body = nn.ModuleList()
        for _ in range(num_blocks):
            self.body.append(ResidualBlock(num_feat))

        # 特征融合
        self.conv_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)

        # 注意力机制 - 增强重要特征
        self.attention = ChannelAttention(num_feat)

        # 重建层
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)

        # 激活函数
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        # 初始化权重
        default_init_weights(
            [self.conv_first, self.conv_body, self.conv_last], 0.1)

    def forward(self, x):
        # 浅层特征
        shallow_feat = self.lrelu(self.conv_first(x))

        # 深层特征提取
        deep_feat = shallow_feat
        for block in self.body:
            deep_feat = block(deep_feat)

        # 特征融合
        body_feat = self.conv_body(deep_feat)
        body_feat = self.lrelu(body_feat + shallow_feat)  # 残差连接

        # 注意力机制
        attended_feat = self.attention(body_feat)

        # 重建
        out = self.conv_last(attended_feat)

        # 全局残差连接 - 保持输入输出一致性
        out = out + x

        return out


class ResidualBlock(nn.Module):
    """残差块"""

    def __init__(self, nf):
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(nf, nf, 3, 1, 1)
        self.conv2 = nn.Conv2d(nf, nf, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        # 初始化
        default_init_weights([self.conv1, self.conv2], 0.1)

    def forward(self, x):
        identity = x
        out = self.lrelu(self.conv1(x))
        out = self.conv2(out)
        return out + identity


class ChannelAttention(nn.Module):
    """通道注意力机制"""

    def __init__(self, nf, reduction=16):
        super(ChannelAttention, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.mlp = nn.Sequential(
            nn.Conv2d(nf, nf // reduction, 1, 1, 0),
            nn.ReLU(inplace=True),
            nn.Conv2d(nf // reduction, nf, 1, 1, 0)
        )

        self.sigmoid = nn.Sigmoid()

        default_init_weights(self.mlp, 0.1)

    def forward(self, x):
        avg_out = self.mlp(self.avg_pool(x))
        max_out = self.mlp(self.max_pool(x))
        attention = self.sigmoid(avg_out + max_out)
        return x * attention


# 轻量级版本 - 如果计算资源有限可以使用这个
@ARCH_REGISTRY.register()
class LightCloudRemovalArch(nn.Module):
    """轻量级云雾去除架构"""

    def __init__(self, num_in_ch=3, num_out_ch=3, num_feat=32, upscale=4):
        super(LightCloudRemovalArch, self).__init__()

        self.conv1 = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        self.conv2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv3 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv4 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)

        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        default_init_weights(
            [self.conv1, self.conv2, self.conv3, self.conv4, self.conv_last], 0.1)

    def forward(self, x):
        feat = self.lrelu(self.conv1(x))
        feat = self.lrelu(self.conv2(feat))
        feat = self.lrelu(self.conv3(feat))
        feat = self.lrelu(self.conv4(feat))

        out = self.conv_last(feat)
        out = out + x  # 残差连接

        return out