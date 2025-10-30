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


class TransmissionEstimator(nn.Module):
    def __init__(self, input=16, groups=4):
        super(TransmissionEstimator, self).__init__()
        self.input = input
        self.groups = groups

        # 调整卷积和池化参数，确保最终输出为[B, 1]
        self.features = nn.Sequential(
            # 输入: [B, 3, H, W]
            nn.Conv2d(3, self.input, kernel_size=5, stride=2),  # 缩小尺寸，减少计算量
            nn.ReLU(),
            # 经过Maxout后通道数变为 groups（原input=16，groups=4 → 16/4=4通道）
            nn.Conv2d(groups, 32, kernel_size=3, padding=1, stride=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),  # 进一步缩小尺寸
            nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))  # 全局池化，固定输出为[B, 64, 1, 1]
        )

        # 最终通过全连接层输出1个值（透射率）
        self.fc = nn.Linear(64, 1)
        self.brelu = BRelu()  # 限制输出在0~1之间

        # 初始化参数
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.normal_(m.weight, mean=0, std=0.001)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def Maxout(self, x, groups):
        # Maxout激活：按组取最大值（将通道数压缩为groups）
        x = x.reshape(x.shape[0], groups, x.shape[1] // groups, x.shape[2], x.shape[3])
        x, _ = torch.max(x, dim=2, keepdim=True)
        return x.reshape(x.shape[0], -1, x.shape[3], x.shape[4])  # 输出通道数=groups

    def forward(self, x):
        # x: [B, 3, H, W]
        out = self.features[0](x)  # 第一层卷积: [B, input, H1, W1]
        out = self.Maxout(out, self.groups)  # 通道数变为groups: [B, groups, H1, W1]

        # 后续特征提取
        for layer in self.features[1:]:  # 从ReLU开始执行剩余层
            out = layer(out)

        # 展平特征并通过全连接层输出透射率
        out = out.view(out.shape[0], -1)  # [B, 64]
        out = self.fc(out)  # [B, 1]
        out = self.brelu(out)  # 限制在0~1之间
        return out  # 输出形状: [B, 1]


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