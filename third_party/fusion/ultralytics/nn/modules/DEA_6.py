# DEA_6.py  (Content-Adaptive Residual-Gated DEA)
# -*- coding: utf-8 -*-

"""
DEA_6 = DEA_5 + 内容自适应 gate（sample-adaptive / content-adaptive）

你当前 DEA_5 的 gate 是：
    g = g_max * sigmoid(alpha)   (alpha 是可学习参数，但与样本无关)

DEA_6 改成：
    g = g_max * sigmoid(alpha + delta(x))
其中 delta(x) 由输入内容自适应生成（根据 RGB 与融合特征的一致性/差异度），从而：
- 对齐好 / 模态一致 -> delta(x) ↑ -> 开门更多
- 对齐差 / 模态不一致 -> delta(x) ↓ -> 更保守，避免污染

保持与 DEA_5 完全兼容的构造参数顺序：
[channel, kernel_size, p_kernel, m_kernel, reduction, g_max, alpha_init, per_channel, use_gate]

推荐用法：
- P3: g_max 很小（0.02） + alpha_init=-6（极保守）
- P5: g_max 更大（0.5） + alpha_init=-2（更愿意用 HSI 语义/材质）

注意：
- 这里 delta(x) 初始为 0（最后一层权重/偏置初始化为 0），所以训练初期等价于 DEA_5（只由 alpha 控制）
- delta(x) 通过轻量 1x1 conv MLP 产生，计算量很小
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

        content-adaptive gate:
            delta = GateNet(rgb, fused)  (sample-adaptive)
            g     = g_max * sigmoid(alpha + delta)

        out = rgb + g * (fused - rgb)
    """

    def __init__(
        self,
        channel=512,
        kernel_size=80,
        p_kernel=None,
        m_kernel=None,
        reduction=16,
        # ---- gate params ----
        g_max=0.30,
        alpha_init=-5.0,
        per_channel=True,
        use_gate=True,
    ):
        super().__init__()

        # Core DEA blocks (same as DEA_5)
        self.deca = DECA(channel, kernel_size, p_kernel, reduction)
        self.depa = DEPA(channel, m_kernel)

        # Output stabilizer (same as DEA_5)
        self.out_conv = Conv(channel, channel, 1, 1)

        # Gate config
        self.use_gate = bool(use_gate)
        self.g_max = float(g_max)
        self.per_channel = bool(per_channel)

        # alpha: learnable base gate logit (same as DEA_5)
        if self.per_channel:
            self.alpha = nn.Parameter(torch.full((1, channel, 1, 1), float(alpha_init)))
        else:
            self.alpha = nn.Parameter(torch.tensor(float(alpha_init)))

        # Content-adaptive delta(x): GateNet
        # Input: [GAP(rgb), GAP(fused), GAP(|rgb-fused|)] -> delta logits
        in_ch = channel * 3
        hidden = max(8, channel // max(1, reduction))  # e.g., 1024//16=64

        if self.per_channel:
            # (B, 3C, 1, 1) -> (B, hidden, 1, 1) -> (B, C, 1, 1)
            self.gate_net = nn.Sequential(
                nn.Conv2d(in_ch, hidden, kernel_size=1, bias=True),
                nn.SiLU(inplace=True),
                nn.Conv2d(hidden, channel, kernel_size=1, bias=True),
            )
        else:
            # scalar gate: (B, 3C, 1, 1) -> (B, hidden, 1, 1) -> (B, 1, 1, 1)
            self.gate_net = nn.Sequential(
                nn.Conv2d(in_ch, hidden, kernel_size=1, bias=True),
                nn.SiLU(inplace=True),
                nn.Conv2d(hidden, 1, kernel_size=1, bias=True),
            )

        # IMPORTANT: make delta(x) start at ~0 so training begins like DEA_5
        self._init_gate_net_zero()

    def _init_gate_net_zero(self):
        # Initialize the LAST conv to zero so delta(x)≈0 initially
        last = None
        for m in self.gate_net.modules():
            if isinstance(m, nn.Conv2d):
                last = m
        if last is not None:
            nn.init.zeros_(last.weight)
            if last.bias is not None:
                nn.init.zeros_(last.bias)

    def _compute_delta(self, rgb, fused):
        # pooled feature statistics: (B,C,1,1)
        rgb_p = F.adaptive_avg_pool2d(rgb, (1, 1))
        fused_p = F.adaptive_avg_pool2d(fused, (1, 1))
        diff_p = F.adaptive_avg_pool2d(torch.abs(rgb - fused), (1, 1))

        gate_in = torch.cat([rgb_p, fused_p, diff_p], dim=1)  # (B,3C,1,1)
        delta = self.gate_net(gate_in)  # (B,C,1,1) or (B,1,1,1)
        return delta

    def forward(self, x):
        rgb, msi = x[0], x[1]

        # DEA core
        deca_out = self.deca([rgb, msi])
        vi, ir = self.depa(deca_out)
        fused = self.out_conv(vi + ir)

        # No gate (ablation)
        if not self.use_gate:
            return fused

        # Content-adaptive gate
        delta = self._compute_delta(rgb, fused)

        # Base alpha broadcast (per_channel) or scalar
        if self.per_channel:
            alpha = self.alpha  # (1,C,1,1)
        else:
            alpha = self.alpha.view(1, 1, 1, 1)  # scalar -> (1,1,1,1)

        g = self.g_max * torch.sigmoid(alpha + delta)  # (B,C,1,1) or (B,1,1,1)

        # Residual-gated fusion
        out = rgb + g * (fused - rgb)
        return out


class DECA(nn.Module):
    """
    DECA (Channel Attention): Soft-Balanced DPSA (Avg + Max)
    - AvgPool captures global/background
    - MaxPool captures sharp texture/highlight
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

        # Early fusion (within attention computation)
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
        x_rgb, x_msi = x[0], x[1]
        b, c, h, w = x_rgb.shape

        x_fuse = self.fusion_conv(torch.cat([x_rgb, x_msi], dim=1))

        # coordinate pooling
        x_h = x_fuse.mean(dim=3, keepdim=True)                       # (b,c,h,1)
        x_w = x_fuse.mean(dim=2, keepdim=True).permute(0, 1, 3, 2)   # (b,c,w,1)

        y = torch.cat([x_h, x_w], dim=2)  # (b,c,h+w,1)
        y = self.act(self.bn1(self.conv1(y)))

        x_hp, x_wp = torch.split(y, [h, w], dim=2)
        x_wp = x_wp.permute(0, 1, 3, 2)  # (b,mip,1,w)

        a_h = self.sigmoid(self.conv_h(x_hp))  # (b,c,h,1)
        a_w = self.sigmoid(self.conv_w(x_wp))  # (b,c,1,w)

        weight = a_h * a_w  # (b,c,h,w)

        out_rgb = x_rgb * weight
        out_msi = x_msi * weight
        return out_rgb, out_msi
