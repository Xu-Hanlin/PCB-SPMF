# DEA.py  (DEA_4 + A2 Residual-Gated Fusion, FIXED)
# -*- coding: utf-8 -*-

"""
DEA (DECA + DEPA) - based on your DEA_4 design, with a SAFE, learnable residual gate.

Why this file?
- Your previous gate g = ReLU(tanh(alpha)) with alpha=0 INIT will "lock" at 0
  because ReLU'(0) ~ 0 -> alpha never updates.
- Here we use: g = g_max * sigmoid(alpha), alpha_init=-5.0 (almost closed but trainable)

Gate:
    out = rgb + g * (fused - rgb)
    g   = g_max * sigmoid(alpha) in (0, g_max)

Notes:
- Keep DEA_4's DECA (Soft-Balanced DPSA) and DEPA (Coordinate Spatial Attention)
- Keep output distribution stable: fused uses a light 1x1 conv (out_conv) before gating
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .conv import Conv


class DEA(nn.Module):
    """
    x[0] -> RGB feature map  (B,C,H,W)
    x[1] -> MSI/HSI feature map (B,C,H,W)

    Forward:
        deca_out = DECA([rgb, msi]) -> [rgb', msi']
        vi, ir   = DEPA(deca_out)
        fused    = out_conv(vi + ir)

        gate g   = g_max * sigmoid(alpha)
        out      = rgb + g * (fused - rgb)
    """

    def __init__(
        self,
        channel=512,
        kernel_size=80,
        p_kernel=None,
        m_kernel=None,
        reduction=16,
        # ---- gate params ----
        g_max=0.30,          # 建议 0.2~0.3；越小越保守
        alpha_init=-5.0,     # 初始强关门，但可学习（sigmoid(-5)≈0.0067）
        per_channel=True,    # True: 每通道门控；False: 一个标量门控
        use_gate=True        # 想做消融可设 False
    ):
        super().__init__()
        self.deca = DECA(channel, kernel_size, p_kernel, reduction)
        self.depa = DEPA(channel, m_kernel)

        self.use_gate = use_gate
        self.g_max = float(g_max)

        # alpha: learnable gate logit
        if per_channel:
            self.alpha = nn.Parameter(torch.full((1, channel, 1, 1), float(alpha_init)))
        else:
            self.alpha = nn.Parameter(torch.tensor(float(alpha_init)))

        # 轻微稳定 fused 的分布（避免 fused-rgb 的数值太野）
        self.out_conv = Conv(channel, channel, 1, 1)

    def forward(self, x):
        rgb, msi = x[0], x[1]

        # DEA core
        deca_out = self.deca([rgb, msi])
        vi, ir = self.depa(deca_out)
        fused = self.out_conv(vi + ir)

        # No gate (ablation)
        if not self.use_gate:
            return fused

        # Safe gate: always differentiable
        g = self.g_max * torch.sigmoid(self.alpha)

        # Residual-gated fusion
        out = rgb + g * (fused - rgb)
        return out


class DECA(nn.Module):
    """
    DECA (Channel Attention): Soft-Balanced DPSA (Avg + Max)
    - AvgPool captures global/background
    - MaxPool captures sharp texture/highlight
    Combine both -> more stable for PCB tiny parts
    """

    def __init__(self, channel=512, kernel_size=80, p_kernel=None, reduction=16):
        super().__init__()
        self.kernel_size = kernel_size

        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        # Shared MLP (no Sigmoid at end here; apply Sigmoid later)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
        )
        self.act = nn.Sigmoid()

        # compress for cross-modal interaction
        self.compress = Conv(channel * 2, channel, 3)

        # convolution pyramid (depthwise)
        if p_kernel is None:
            p_kernel = [5, 4]
        k1, k2 = p_kernel

        self.conv_c1 = nn.Sequential(
            nn.Conv2d(channel, channel, k1, k1, 0, groups=channel),
            nn.SiLU()
        )
        self.conv_c2 = nn.Sequential(
            nn.Conv2d(channel, channel, k2, k2, 0, groups=channel),
            nn.SiLU()
        )
        k3 = int(self.kernel_size / k1 / k2)
        k3 = max(1, k3)
        self.conv_c3 = nn.Sequential(
            nn.Conv2d(channel, channel, k3, k3, 0, groups=channel),
            nn.SiLU()
        )

    def _weight_from_pool(self, pooled_vec):
        # pooled_vec: (b, c)
        b, c = pooled_vec.shape
        w = self.fc(pooled_vec).view(b, c, 1, 1)
        return self.act(w)

    def forward(self, x):
        # x = [rgb, msi]
        rgb, msi = x[0], x[1]
        b, c, h, w = rgb.shape

        # Soft-Balanced weights: avg + max
        avg_rgb = self.avg_pool(rgb).view(b, c)
        max_rgb = rgb.view(b, c, -1).max(dim=2)[0]
        w_rgb = 0.5 * (self._weight_from_pool(avg_rgb) + self._weight_from_pool(max_rgb))

        avg_msi = self.avg_pool(msi).view(b, c)
        max_msi = msi.view(b, c, -1).max(dim=2)[0]
        w_msi = 0.5 * (self._weight_from_pool(avg_msi) + self._weight_from_pool(max_msi))

        # Cross-modal interaction
        glob_t = self.compress(torch.cat([rgb, msi], dim=1))

        if min(h, w) >= self.kernel_size:
            glob = self.conv_c3(self.conv_c2(self.conv_c1(glob_t)))
            # if not 1x1, pool to 1x1 for stability
            if glob.shape[-2] != 1 or glob.shape[-1] != 1:
                glob = F.adaptive_avg_pool2d(glob, (1, 1))
        else:
            glob = torch.mean(glob_t, dim=[2, 3], keepdim=True)

        # Cross-guided reweighting
        rgb_out = rgb * (self.act(w_msi * glob)).expand_as(rgb)
        msi_out = msi * (self.act(w_rgb * glob)).expand_as(msi)

        return [rgb_out, msi_out]


class DEPA(nn.Module):
    """
    DEPA (Spatial Attention): Coordinate Spatial Attention (DCSA)
    - better for elongated structures (connector)
    """

    def __init__(self, channel=512, m_kernel=None, reduction=32):
        super().__init__()

        # Early fusion
        self.fusion_conv = Conv(channel * 2, channel, 1, 1)

        # Coordinate attention bottleneck
        mip = max(8, channel // reduction)
        self.conv1 = nn.Conv2d(channel, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = nn.SiLU(inplace=True)

        # projections for H and W
        self.conv_h = nn.Conv2d(mip, channel, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, channel, kernel_size=1, stride=1, padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x = [rgb', msi']
        x_rgb, x_msi = x[0], x[1]
        b, c, h, w = x_rgb.shape

        x_fuse = self.fusion_conv(torch.cat([x_rgb, x_msi], dim=1))

        # coordinate pooling
        x_h = x_fuse.mean(dim=3, keepdim=True)                 # (b,c,h,1)
        x_w = x_fuse.mean(dim=2, keepdim=True).permute(0, 1, 3, 2)  # (b,c,w,1)

        y = torch.cat([x_h, x_w], dim=2)  # (b,c,h+w,1)
        y = self.act(self.bn1(self.conv1(y)))

        x_hp, x_wp = torch.split(y, [h, w], dim=2)
        x_wp = x_wp.permute(0, 1, 3, 2)   # (b,mip,1,w)

        a_h = self.sigmoid(self.conv_h(x_hp))  # (b,c,h,1)
        a_w = self.sigmoid(self.conv_w(x_wp))  # (b,c,1,w)

        weight = a_h * a_w  # (b,c,h,w) broadcast

        out_rgb = x_rgb * weight
        out_msi = x_msi * weight
        return out_rgb, out_msi
