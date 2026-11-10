import torch
import torch.nn as nn
from ultralytics.nn.modules.conv import Conv

class BandSelectionModule(nn.Module):
    """
    自适应波段选择模块
    输入: MSI/HyperSpectral 数据 (B, Cin, H, W)
    输出: 经过权重调整后的特征图，或降维后的特征图
    """
    def __init__(self, c1, c2=None):
        super().__init__()
        if c2 is None: c2 = c1
        # 简单的通道注意力机制 (类似 SE Block)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(c1, c1 // 2 if c1 > 1 else 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(c1 // 2 if c1 > 1 else 1, c1, bias=False),
            nn.Sigmoid()
        )
        # 如果需要调整通道数（降维），加一个卷积
        self.reduce = Conv(c1, c2, 1, 1) if c1 != c2 else nn.Identity()

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return self.reduce(x * y.expand_as(x))