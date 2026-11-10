# DEA_7.py
# Ultralytics-compatible DEA module (RGB+MSI/HSI)
# Key design goals:
# 1) Entry Gate (alpha): clean MSI before it participates anywhere (reduce misalignment/noise pollution).
# 2) Mix Gate  (beta): control how much "fused enhancement" is added back to RGB (avoid gate^2 vanishing).
# 3) Sample-wise Modality Dropout: during training, randomly drop MSI PER SAMPLE; dropped samples become STRICT RGB passthrough.
# 4) Inference safety: if MSI is missing / nearly zero, bypass to strict RGB.
# 5) Small-batch stability: GroupNorm in DEPA (instead of BN).

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.nn.modules.conv import Conv


class DEA(nn.Module):
    """
    DEA - Robust Split-Gate Version

    YAML args order (compatible with your previous usage):
    [channel, kernel_size, p_kernel, m_kernel, reduction, g_max, alpha_init, per_channel, use_gate]

    Extra (optional) args:
    - dropout: sample-wise modality dropout probability during training (default 0.2)
    - beta_init: init value for mix gate (default -5.0)
    - bypass_eps_gate: inference bypass threshold on gate_in mean
    - bypass_eps_msi:  inference bypass threshold on MSI magnitude
    """

    def __init__(
        self,
        channel=512,
        kernel_size=80,
        p_kernel=None,
        m_kernel=None,
        reduction=16,
        g_max=1.0,
        alpha_init=-5.0,
        per_channel=True,
        use_gate=True,
        dropout=0.2,
        beta_init=-5.0,
        bypass_eps_gate=1e-5,
        bypass_eps_msi=1e-8,
    ):
        super().__init__()
        self.deca = DECA(channel, kernel_size, p_kernel, reduction)
        self.depa = DEPA(channel, m_kernel)

        self.use_gate = use_gate
        self.g_max = float(g_max)
        self.dropout = float(dropout)
        self.bypass_eps_gate = float(bypass_eps_gate)
        self.bypass_eps_msi = float(bypass_eps_msi)

        if self.use_gate:
            # Entry gate alpha (can be per-channel)
            shape = (1, channel, 1, 1) if per_channel else (1,)
            self.alpha = nn.Parameter(torch.full(shape, float(alpha_init)))

            # Mix gate beta (scalar, per-DEA instance). Controls residual mixing strength.
            self.beta = nn.Parameter(torch.tensor([float(beta_init)]))

            self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: [rgb_feat, msi_feat]
        rgb, msi = x[0], x[1]
        B = rgb.shape[0]

        if self.use_gate:
            # Entry gate: (0..g_max)
            gate_in = self.sigmoid(self.alpha) * self.g_max  # (1,C,1,1) or (1,)
            # Mix gate: (0..1), scalar
            mix_weight = self.sigmoid(self.beta)             # (1,)

            # --- Sample-wise Modality Dropout (TRAIN ONLY) ---
            # Drop MSI per-sample; dropped samples MUST become strict RGB passthrough.
            if self.training and self.dropout > 0.0:
                keep_mask = (torch.rand(B, 1, 1, 1, device=rgb.device) >= self.dropout).to(rgb.dtype)
                gate_in = gate_in * keep_mask
                # CRITICAL: also mask mix_weight so dropped samples are EXACT rgb passthrough.
                mix_weight = mix_weight * keep_mask

            # --- Inference bypass (STRICT RGB) ---
            # If MSI is absent or gate is ~0, return rgb exactly.
            if (not self.training):
                # 1) gate-based bypass
                # gate_in may be broadcastable; mean() is safe.
                if gate_in.mean().item() < self.bypass_eps_gate:
                    return rgb
                # 2) MSI-magnitude-based bypass (anti "msi all-zeros but gate nonzero" corner case)
                if msi.abs().mean().item() < self.bypass_eps_msi:
                    return rgb

        else:
            # Gate disabled: behave like always-on fusion (not recommended for your misalignment setting)
            gate_in = rgb.new_tensor(1.0)
            mix_weight = rgb.new_tensor(1.0)

        # --- Entry gating (clean MSI before any attention/fusion) ---
        msi_eff = msi * gate_in

        # --- Attention blocks ---
        # Note: even if msi_eff=0, DECA/DEPA computes a deterministic "RGB self-attention" style transform.
        deca_out = self.deca([rgb, msi_eff])
        result_vi, result_ir = self.depa(deca_out)

        fused_rgb = result_vi + result_ir

        # --- Residual mixing (prevents gate^2 vanishing) ---
        # Out = RGB + mix * (Fused - RGB)
        # If mix=0 (e.g., dropout sample), output is STRICT rgb.
        out = rgb + mix_weight * (fused_rgb - rgb)
        return out


class DECA(nn.Module):
    """
    DECA: Channel attention with cross-modal interaction.
    When MSI is gated to zero, it reduces to an RGB-determined modulation (still stable).
    """

    def __init__(self, channel=512, kernel_size=80, p_kernel=None, reduction=16):
        super().__init__()
        self.kernel_size = int(kernel_size)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # MLP without bias (keeps 0->0, stable when MSI=0)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
        )
        self.act = nn.Sigmoid()
        self.compress = Conv(channel * 2, channel, 3)

        # Convolution pyramid (depthwise)
        if p_kernel is None:
            p_kernel = [5, 4]
        k1, k2 = p_kernel
        self.conv_c1 = nn.Sequential(nn.Conv2d(channel, channel, k1, k1, 0, groups=channel), nn.SiLU())
        self.conv_c2 = nn.Sequential(nn.Conv2d(channel, channel, k2, k2, 0, groups=channel), nn.SiLU())
        k3 = max(1, int(self.kernel_size / max(1, k1) / max(1, k2)))
        self.conv_c3 = nn.Sequential(nn.Conv2d(channel, channel, k3, k3, 0, groups=channel), nn.SiLU())

    def _get_weight(self, feat_1d):
        # feat_1d: (B,C)
        return self.act(self.fc(feat_1d))

    def forward(self, x):
        rgb, msi = x[0], x[1]
        b, c, h, w = rgb.size()

        # RGB weights
        avg_vi = self.avg_pool(rgb).view(b, c)
        max_vi = rgb.view(b, c, -1).max(dim=2)[0]
        w_vi = 0.5 * (self._get_weight(avg_vi) + self._get_weight(max_vi))
        w_vi = w_vi.view(b, c, 1, 1)

        # MSI weights
        avg_ir = self.avg_pool(msi).view(b, c)
        max_ir = msi.view(b, c, -1).max(dim=2)[0]
        w_ir = 0.5 * (self._get_weight(avg_ir) + self._get_weight(max_ir))
        w_ir = w_ir.view(b, c, 1, 1)

        # Cross interaction global descriptor
        glob_t = self.compress(torch.cat([rgb, msi], dim=1))
        if min(h, w) >= self.kernel_size:
            glob = self.conv_c3(self.conv_c2(self.conv_c1(glob_t)))
            if glob.shape[-2:] != (1, 1):
                glob = F.adaptive_avg_pool2d(glob, (1, 1))
        else:
            glob = torch.mean(glob_t, dim=[2, 3], keepdim=True)

        # Cross modulation
        result_vi = rgb * self.act(w_ir * glob).expand_as(rgb)
        result_ir = msi * self.act(w_vi * glob).expand_as(msi)
        return [result_vi, result_ir]


class DEPA(nn.Module):
    """
    DEPA: Coordinate Spatial Attention (shared spatial weight for both modalities)
    Uses GroupNorm for small batch stability.
    """

    def __init__(self, channel=512, m_kernel=None, reduction=32):
        super().__init__()

        # Early fusion inside attention computation
        self.fusion_conv = Conv(channel * 2, channel, 1, 1)

        mip = max(8, channel // reduction)
        self.conv1 = nn.Conv2d(channel, mip, kernel_size=1, stride=1, padding=0)

        # GroupNorm is stable for B=1~8 (typical detection training)
        # Use 1 group if mip < 8; otherwise 8 groups (safe default)
        g = 8 if mip >= 8 else 1
        self.gn1 = nn.GroupNorm(num_groups=g, num_channels=mip)

        self.act = nn.SiLU(inplace=True)

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
        y = self.act(self.gn1(self.conv1(y)))

        x_hp, x_wp = torch.split(y, [h, w], dim=2)
        x_wp = x_wp.permute(0, 1, 3, 2)  # (b,mip,1,w)

        a_h = self.sigmoid(self.conv_h(x_hp))  # (b,c,h,1)
        a_w = self.sigmoid(self.conv_w(x_wp))  # (b,c,1,w)

        weight = a_h * a_w  # (b,c,h,w)

        out_rgb = x_rgb * weight
        out_msi = x_msi * weight
        return out_rgb, out_msi
