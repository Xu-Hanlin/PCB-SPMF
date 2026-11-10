import torch
import torch.nn as nn
import torch.nn.functional as F

class RFF_Lite(nn.Module):
    def __init__(self, c_p1, c_p3, amax=0.2):
        super().__init__()
        self.down = nn.Sequential(
            nn.Conv2d(c_p1, c_p3, kernel_size=3, stride=4, padding=1, bias=False),
            nn.GroupNorm(num_groups=self._pick_groups(c_p3), num_channels=c_p3),
            nn.SiLU(inplace=True),
        )
        self.alpha_raw = nn.Parameter(torch.tensor([-6.0]))
        self.amax = float(amax)

    @staticmethod
    def _pick_groups(ch):
        for g in (32, 16, 8, 4, 2, 1):
            if ch % g == 0:
                return g
        return 1

    def forward(self, x):
        if not (isinstance(x, (list, tuple)) and len(x) >= 3):
            return x

        p1_rgb = x[0]
        p3 = x[2]

        p1_inject = self.down(p1_rgb)
        if p1_inject.shape[-2:] != p3.shape[-2:]:
            p1_inject = F.interpolate(p1_inject, size=p3.shape[-2:], mode="nearest")

        alpha = (torch.sigmoid(self.alpha_raw) * self.amax).to(dtype=p3.dtype, device=p3.device)
        return p3 + p1_inject * alpha
