import torch
import torch.nn as nn

class RFF_Lite(nn.Module):
    """
    RFF-Lite (Safe Mode):
    - 输入: [P1_rgb, P1_hsi, P3_rgb]  (注意：来自 YAML 的 from 是 list，Ultralytics 会传 list 进来)
    - 只用 P1_rgb 做细节回流，忽略 P1_hsi（防止噪声污染）
    - stride=4: 将 P1(H/2) 对齐到 P3(H/8)
    - alpha 从 0 开始，训练初期等价于不开启回流，稳定
    """
    def __init__(self, c_p1: int, c_p3: int):
        super().__init__()
        self.down = nn.Sequential(
            nn.Conv2d(c_p1, c_p3, kernel_size=3, stride=4, padding=1, bias=False),
            nn.BatchNorm2d(c_p3),
            nn.SiLU(inplace=True),
        )
        self.alpha = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        # x 是一个 list/tuple: [p1_rgb, p1_hsi, p3]
        p1_rgb, _p1_hsi, p3 = x
        p1_inject = self.down(p1_rgb)
        return p3 + p1_inject * self.alpha
