import torch
from torch import nn as nn
from torch.nn import functional as F  # noqa: F401

from basicsr.archs.arch_util import default_init_weights
from basicsr.utils.registry import ARCH_REGISTRY


# 边界ReLU激活函数（适配GPU）
class BRelu(nn.Hardtanh):
    def __init__(self, inplace=False):
        super(BRelu, self).__init__(0., 1., inplace)

    def extra_repr(self):
        inplace_str = 'inplace=True' if self.inplace else ''
        return inplace_str


@ARCH_REGISTRY.register()
class DehazeNet(nn.Module):
    def __init__(self, input=16, groups=4):
        super(DehazeNet, self).__init__()
        self.input = input
        self.groups = groups

        # 调整卷积层参数以保持尺寸一致
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=self.input, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(in_channels=4, out_channels=16, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=4, out_channels=16, kernel_size=5, padding=2)
        self.conv4 = nn.Conv2d(in_channels=4, out_channels=16, kernel_size=7, padding=3)
        self.maxpool = nn.MaxPool2d(kernel_size=7, stride=1, padding=3)
        self.conv5 = nn.Conv2d(in_channels=48, out_channels=3, kernel_size=6, padding=3)
        self.brelu = BRelu()

        # 初始化权重
        for name, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0, std=0.001)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def Maxout(self, x, groups):
        x = x.reshape(x.shape[0], groups, x.shape[1] // groups, x.shape[2], x.shape[3])
        x, _ = torch.max(x, dim=2, keepdim=True)
        out = x.reshape(x.shape[0], -1, x.shape[3], x.shape[4])
        return out

    def forward(self, x):
        # 保存输入尺寸用于最终调整
        input_h, input_w = x.shape[2], x.shape[3]

        # 特征提取与处理
        out = self.conv1(x)
        out = self.Maxout(out, self.groups)

        out1 = self.conv2(out)
        out2 = self.conv3(out)
        out3 = self.conv4(out)

        y = torch.cat((out1, out2, out3), dim=1)
        y = self.maxpool(y)
        y = self.conv5(y)
        y = self.brelu(y)

        # 确保输出尺寸与输入完全一致（处理可能的尺寸偏差）
        if y.shape[2] != input_h or y.shape[3] != input_w:
            y = F.interpolate(y, size=(input_h, input_w), mode='bilinear', align_corners=False)

        return y

