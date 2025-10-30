import torch
from torch import nn as nn
from torch.nn import functional as F

from basicsr.archs.arch_util import default_init_weights
from basicsr.utils.registry import ARCH_REGISTRY


# 自定义BRelu激活函数（修正原代码中的大小写问题）
class BRelu(nn.Hardtanh):
    def __init__(self, inplace=False):
        super(BRelu, self).__init__(0., 1., inplace)  # 限制输出在0~1之间

    def extra_repr(self):
        inplace_str = 'inplace=True' if self.inplace else ''
        return inplace_str


# 透射率估计器（作为内部组件）
class TransmissionEstimator(nn.Module):
    def __init__(self, input=16, groups=4):
        super(TransmissionEstimator, self).__init__()
        self.input = input
        self.groups = groups
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=self.input, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=4, out_channels=16, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=4, out_channels=16, kernel_size=5, padding=2)
        self.conv4 = nn.Conv2d(in_channels=4, out_channels=16, kernel_size=7, padding=3)
        self.maxpool = nn.MaxPool2d(kernel_size=7, stride=1)
        self.conv5 = nn.Conv2d(in_channels=48, out_channels=1, kernel_size=6)
        self.brelu = BRelu()  # 使用自定义的BRelu（修正原代码的nn.BReLU错误）

        # 初始化卷积层参数
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0, std=0.001)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def Maxout(self, x, groups):
        # Maxout激活：按组取最大值
        x = x.reshape(x.shape[0], groups, x.shape[1] // groups, x.shape[2], x.shape[3])
        x, _ = torch.max(x, dim=2, keepdim=True)
        return x.reshape(x.shape[0], -1, x.shape[3], x.shape[4])

    def forward(self, x):
        out = self.conv1(x)
        out = self.Maxout(out, self.groups)
        out1 = self.conv2(out)
        out2 = self.conv3(out)
        out3 = self.conv4(out)
        y = torch.cat((out1, out2, out3), dim=1)
        y = self.maxpool(y)
        y = self.conv5(y)
        y = self.brelu(y)
        return y.reshape(y.shape[0], -1)  # 返回[B, 1]形状的透射率初始估计


@ARCH_REGISTRY.register()
class DehazeNet(nn.Module):
    # 合并模型：直接在DehazeNet内部定义transmission_estimator
    def __init__(self, trans_input=16, trans_groups=4):  # 暴露透射率估计器的参数
        super(DehazeNet, self).__init__()

        # 1. 内部实例化透射率估计器（不再依赖外部传入）
        self.transmission_estimator = TransmissionEstimator(
            input=trans_input,
            groups=trans_groups
        )

        # 2. 大气光估计网络
        self.atmospheric_light_net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(3, 16, 1),
            nn.ReLU(),
            nn.Conv2d(16, 3, 1),
            nn.Sigmoid()
        )

        # 3. 透射率细化网络
        self.transmission_refine = nn.Sequential(
            nn.Conv2d(4, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 1, 3, padding=1),
            nn.Sigmoid()
        )

        # 4. 图像恢复网络
        self.image_recovery = nn.Sequential(
            nn.Conv2d(4, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 3, 3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, cloudy_img):
        # cloudy_img: [B, 3, H, W]（输入带雾/云的图像）

        # 1. 估计初始透射率（使用内部的transmission_estimator）
        with torch.no_grad():  # 不计算该部分梯度（可选，根据需求调整）
            batch_size, _, h, w = cloudy_img.shape
            transmission = self.transmission_estimator(cloudy_img)  # [B, 1]
            transmission = transmission.view(batch_size, 1, 1, 1).expand(-1, -1, h, w)  # 扩展为[B, 1, H, W]

        # 2. 估计大气光
        atmospheric_light = self.atmospheric_light_net(cloudy_img)  # [B, 3, 1, 1]

        # 3. 细化透射率图
        transmission_input = torch.cat([cloudy_img, transmission], dim=1)  # 拼接输入图像和初始透射率
        refined_transmission = self.transmission_refine(transmission_input)  # [B, 1, H, W]

        # 4. 应用大气散射模型得到初步去云结果
        transmission_clamped = torch.clamp(refined_transmission, 0.1, 0.99)  # 限制透射率范围，避免除以0
        preliminary_result = (cloudy_img - atmospheric_light * (1 - transmission_clamped)) / transmission_clamped
        preliminary_result = torch.clamp(preliminary_result, 0, 1)  # 限制在0~1之间

        # 5. 使用CNN进一步恢复图像细节
        recovery_input = torch.cat([preliminary_result, refined_transmission], dim=1)
        final_result = self.image_recovery(recovery_input)  # [B, 3, H, W]（输出去雾/云后的图像）

        return final_result