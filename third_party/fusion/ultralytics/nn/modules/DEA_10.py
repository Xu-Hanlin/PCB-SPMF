# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.nn.modules.conv import Conv


class LearnableWarp(nn.Module):
    def __init__(self, max_shift=1.0, align_corners=False):
        super().__init__()
        self.max_shift = float(max_shift)
        self.align_corners = bool(align_corners)

    @staticmethod
    def _make_base_grid(b, h, w, device, dtype):
        ys = torch.linspace(-1.0, 1.0, h, device=device, dtype=dtype)
        xs = torch.linspace(-1.0, 1.0, w, device=device, dtype=dtype)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
        base = torch.stack([grid_x, grid_y], dim=-1)  # (H,W,2)
        base = base.unsqueeze(0).expand(b, h, w, 2).contiguous()  # (B,H,W,2)
        return base

    def forward(self, rgb, msi, dx, dy):
        b, _, h, w = rgb.shape
        base = self._make_base_grid(b, h, w, rgb.device, rgb.dtype)

        dx = torch.tanh(dx) * self.max_shift
        dy = torch.tanh(dy) * self.max_shift

        dx = dx.view(b, 1, 1)
        dy = dy.view(b, 1, 1)

        dx_norm = dx * (2.0 / max(w - 1, 1))
        dy_norm = dy * (2.0 / max(h - 1, 1))

        base_x = base[..., 0] + dx_norm
        base_y = base[..., 1] + dy_norm
        grid = torch.stack([base_x, base_y], dim=-1)

        out = F.grid_sample(
            msi, grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=self.align_corners
        )
        return out


class DEA(nn.Module):
    """
    Args order:
    [channel, kernel_size, p_kernel, m_kernel, reduction, g_max, alpha_init, per_channel, use_gate]

    Extra args:
    dropout: float
    beta_init: float
    bypass_eps_gate: float
    bypass_eps_msi: float
    warp: bool
    warp_max_shift: float
    warp_align_corners: bool
    warp_init: float
    warp_learnable: bool
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
        warp=True,
        warp_max_shift=1.0,
        warp_align_corners=False,
        warp_init=0.0,
        warp_learnable=True,
    ):
        super().__init__()
        self.deca = DECA(channel, kernel_size, p_kernel, reduction)
        self.depa = DEPA(channel, m_kernel)

        self.use_gate = bool(use_gate)
        self.g_max = float(g_max)
        self.dropout = float(dropout)
        self.bypass_eps_gate = float(bypass_eps_gate)
        self.bypass_eps_msi = float(bypass_eps_msi)

        self.warp_on = bool(warp)
        self.warp = LearnableWarp(max_shift=warp_max_shift, align_corners=warp_align_corners)

        self.sigmoid = nn.Sigmoid()

        if self.use_gate:
            shape = (1, channel, 1, 1) if per_channel else (1,)
            self.alpha = nn.Parameter(torch.full(shape, float(alpha_init)))
            self.beta = nn.Parameter(torch.tensor([float(beta_init)]))

        if self.warp_on:
            self.warp_learnable = bool(warp_learnable)
            if self.warp_learnable:
                self.warp_pool = nn.AdaptiveAvgPool2d(1)
                self.warp_fc = nn.Sequential(
                    nn.Linear(channel, max(8, channel // 32), bias=True),
                    nn.SiLU(inplace=True),
                    nn.Linear(max(8, channel // 32), 2, bias=True),
                )
                nn.init.zeros_(self.warp_fc[-1].weight)
                nn.init.constant_(self.warp_fc[-1].bias, float(warp_init))
            else:
                self.register_buffer("warp_dx_const", torch.tensor([float(warp_init)]), persistent=False)
                self.register_buffer("warp_dy_const", torch.tensor([float(warp_init)]), persistent=False)

    def _samplewise_dropout(self, b, device, dtype):
        keep = (torch.rand(b, 1, 1, 1, device=device) >= self.dropout).to(dtype)
        return keep

    def forward(self, x):
        rgb, msi = x[0], x[1]
        b = rgb.shape[0]

        if self.use_gate:
            gate_in = self.sigmoid(self.alpha) * self.g_max
            mix_weight = self.sigmoid(self.beta)

            if self.training and self.dropout > 0.0:
                keep_mask = self._samplewise_dropout(b, rgb.device, rgb.dtype)
                gate_in = gate_in * keep_mask
                mix_weight = mix_weight * keep_mask

            if not self.training:
                if gate_in.mean().item() < self.bypass_eps_gate:
                    return rgb
                if msi.abs().mean().item() < self.bypass_eps_msi:
                    return rgb
        else:
            gate_in = rgb.new_tensor(1.0)
            mix_weight = rgb.new_tensor(1.0)

        msi_eff = msi * gate_in

        if self.warp_on:
            if self.warp_learnable:
                z = self.warp_pool(rgb).flatten(1)
                dxy = self.warp_fc(z)  # (B,2)
                dx = dxy[:, 0]
                dy = dxy[:, 1]
            else:
                dx = self.warp_dx_const.expand(b)
                dy = self.warp_dy_const.expand(b)

            msi_eff = self.warp(rgb, msi_eff, dx, dy)

        deca_out = self.deca([rgb, msi_eff])
        out_rgb, out_msi = self.depa(deca_out)

        fused_rgb = out_rgb + out_msi

        out = rgb + mix_weight * (fused_rgb - rgb)
        return out


class DECA(nn.Module):
    def __init__(self, channel=512, kernel_size=80, p_kernel=None, reduction=16):
        super().__init__()
        self.kernel_size = int(kernel_size)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
        )
        self.act = nn.Sigmoid()
        self.compress = Conv(channel * 2, channel, 3)

        if p_kernel is None:
            p_kernel = [5, 4]
        k1, k2 = p_kernel
        self.conv_c1 = nn.Sequential(nn.Conv2d(channel, channel, k1, k1, 0, groups=channel), nn.SiLU())
        self.conv_c2 = nn.Sequential(nn.Conv2d(channel, channel, k2, k2, 0, groups=channel), nn.SiLU())
        k3 = max(1, int(self.kernel_size / max(1, k1) / max(1, k2)))
        self.conv_c3 = nn.Sequential(nn.Conv2d(channel, channel, k3, k3, 0, groups=channel), nn.SiLU())

    def _get_weight(self, feat_1d):
        return self.act(self.fc(feat_1d))

    def forward(self, x):
        rgb, msi = x[0], x[1]
        b, c, h, w = rgb.size()

        avg_vi = self.avg_pool(rgb).view(b, c)
        max_vi = rgb.view(b, c, -1).max(dim=2)[0]
        w_vi = 0.5 * (self._get_weight(avg_vi) + self._get_weight(max_vi))
        w_vi = w_vi.view(b, c, 1, 1)

        avg_ir = self.avg_pool(msi).view(b, c)
        max_ir = msi.view(b, c, -1).max(dim=2)[0]
        w_ir = 0.5 * (self._get_weight(avg_ir) + self._get_weight(max_ir))
        w_ir = w_ir.view(b, c, 1, 1)

        glob_t = self.compress(torch.cat([rgb, msi], dim=1))
        if min(h, w) >= self.kernel_size:
            glob = self.conv_c3(self.conv_c2(self.conv_c1(glob_t)))
            if glob.shape[-2:] != (1, 1):
                glob = F.adaptive_avg_pool2d(glob, (1, 1))
        else:
            glob = torch.mean(glob_t, dim=[2, 3], keepdim=True)

        result_vi = rgb * self.act(w_ir * glob).expand_as(rgb)
        result_ir = msi * self.act(w_vi * glob).expand_as(msi)
        return [result_vi, result_ir]


class DEPA(nn.Module):
    def __init__(self, channel=512, m_kernel=None, reduction=32):
        super().__init__()

        self.fusion_conv = Conv(channel * 2, channel, 1, 1)

        mip = max(8, channel // reduction)
        self.conv1 = nn.Conv2d(channel, mip, kernel_size=1, stride=1, padding=0)

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

        x_h = x_fuse.mean(dim=3, keepdim=True)
        x_w = x_fuse.mean(dim=2, keepdim=True).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.act(self.gn1(self.conv1(y)))

        x_hp, x_wp = torch.split(y, [h, w], dim=2)
        x_wp = x_wp.permute(0, 1, 3, 2)

        a_h = self.sigmoid(self.conv_h(x_hp))
        a_w = self.sigmoid(self.conv_w(x_wp))

        weight = a_h * a_w

        out_rgb = x_rgb * weight
        out_msi = x_msi * weight
        return out_rgb, out_msi
