import torch
from torch import nn as nn
from torch.nn import functional as F  # noqa: F401

from basicsr.archs.arch_util import default_init_weights
from basicsr.utils.registry import ARCH_REGISTRY

@ARCH_REGISTRY.register()
class DehazeNet(nn.Module):
    def __init__(self, transmission_estimator):
        super(DehazeNet, self).__init__()
        self.transmission_estimator = transmission_estimator

        # 大气光估计网络
        self.atmospheric_light_net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(3, 16, 1),
            nn.ReLU(),
            nn.Conv2d(16, 3, 1),
            nn.Sigmoid()
        )

        # 透射率细化网络
        self.transmission_refine = nn.Sequential(
            nn.Conv2d(4, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 1, 3, padding=1),
            nn.Sigmoid()
        )

        # 图像恢复网络
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
        # cloudy_img: [B, 3, H, W]

        # 1. 估计初始透射率
        with torch.no_grad():
            batch_size, _, h, w = cloudy_img.shape
            transmission = self.transmission_estimator(cloudy_img)  # [B, 1]
            transmission = transmission.view(batch_size, 1, 1, 1).expand(-1, -1, h, w)

        # 2. 估计大气光
        atmospheric_light = self.atmospheric_light_net(cloudy_img)  # [B, 3, 1, 1]

        # 3. 细化透射率图
        transmission_input = torch.cat([cloudy_img, transmission], dim=1)
        refined_transmission = self.transmission_refine(transmission_input)  # [B, 1, H, W]

        # 4. 应用大气散射模型得到初步去云结果
        transmission_clamped = torch.clamp(refined_transmission, 0.1, 0.99)
        preliminary_result = (cloudy_img - atmospheric_light * (1 - transmission_clamped)) / transmission_clamped
        preliminary_result = torch.clamp(preliminary_result, 0, 1)

        # 5. 使用CNN进一步恢复图像细节
        recovery_input = torch.cat([preliminary_result, refined_transmission], dim=1)
        final_result = self.image_recovery(recovery_input)  # [B, 3, H, W]

        return final_result

