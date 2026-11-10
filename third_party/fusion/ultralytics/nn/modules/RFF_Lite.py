#1
import torch
import torch.nn as nn

class RFF_Lite(nn.Module):

    def __init__(self, c_p1: int, c_p3: int):
        super().__init__()
        self.down = nn.Sequential(
            nn.Conv2d(c_p1, c_p3, kernel_size=3, stride=4, padding=1, bias=False),
            nn.BatchNorm2d(c_p3),
            nn.SiLU(inplace=True),
        )
        self.alpha = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        p1_rgb, _p1_hsi, p3 = x
        p1_inject = self.down(p1_rgb)
        return p3 + p1_inject * self.alpha
