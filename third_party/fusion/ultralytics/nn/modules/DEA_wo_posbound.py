# -*- coding: utf-8 -*-
"""
DEA ablation: w/o positive bound

Change:
- remove positive-only global gain constraint
- global gain becomes signed: gain = mod_cap * tanh(mod_gain)

All other parts are kept the same.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.nn.modules.conv import Conv


class LearnableWarp(nn.Module):
    def __init__(self, max_shift=1.5, align_corners=False):
        super().__init__()
        self.max_shift = float(max_shift)
        self.align_corners = bool(align_corners)

    @staticmethod
    def _make_base_grid(b, h, w, device, dtype):
        ys = torch.linspace(-1.0, 1.0, h, device=device, dtype=dtype)
        xs = torch.linspace(-1.0, 1.0, w, device=device, dtype=dtype)
        gy, gx = torch.meshgrid(ys, xs, indexing="ij")
        base = torch.stack([gx, gy], dim=-1)
        return base.unsqueeze(0).expand(b, h, w, 2).contiguous()

    def forward(self, ref, src, dx, dy):
        b, _, h, w = ref.shape
        base = self._make_base_grid(b, h, w, ref.device, ref.dtype)

        dx = torch.tanh(dx) * self.max_shift
        dy = torch.tanh(dy) * self.max_shift

        dx_norm = dx.view(b, 1, 1) * (2.0 / max(w - 1, 1))
        dy_norm = dy.view(b, 1, 1) * (2.0 / max(h - 1, 1))

        grid_x = base[..., 0] + dx_norm
        grid_y = base[..., 1] + dy_norm
        grid = torch.stack([grid_x, grid_y], dim=-1)

        return F.grid_sample(
            src,
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=self.align_corners,
        )


class LocalModulationGate(nn.Module):
    def __init__(self, reduction=16, init=-1.1, floor=0.05, temp=1.5, sparse_pow=2.0):
        super().__init__()
        mid = max(8, 64 // max(1, int(reduction)))
        groups = 4 if mid % 4 == 0 else 1

        self.net = nn.Sequential(
            nn.Conv2d(8, mid, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(groups, mid),
            nn.SiLU(inplace=True),
            nn.Conv2d(mid, mid, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(groups, mid),
            nn.SiLU(inplace=True),
            nn.Conv2d(mid, 1, kernel_size=1, stride=1, padding=0, bias=True),
        )
        nn.init.constant_(self.net[-1].bias, float(init))

        self.floor = float(floor)
        self.temp = float(temp)
        self.sparse_pow = float(sparse_pow)

    @staticmethod
    def _avgmax(x):
        return x.mean(dim=1, keepdim=True), x.amax(dim=1, keepdim=True)

    @staticmethod
    def _local_contrast(x):
        base = x.mean(dim=1, keepdim=True)
        smooth = F.avg_pool2d(base, kernel_size=3, stride=1, padding=1)
        return torch.abs(base - smooth)

    @staticmethod
    def _edge_proxy(x):
        base = x.mean(dim=1, keepdim=True)
        gx = torch.abs(base[:, :, :, 1:] - base[:, :, :, :-1])
        gy = torch.abs(base[:, :, 1:, :] - base[:, :, :-1, :])
        gx = F.pad(gx, (0, 1, 0, 0))
        gy = F.pad(gy, (0, 0, 0, 1))
        return gx + gy

    def forward(self, rgb, msi_eff):
        diff = torch.abs(rgb - msi_eff)

        rgb_avg, rgb_max = self._avgmax(rgb)
        diff_avg, diff_max = self._avgmax(diff)
        rgb_loc = self._local_contrast(rgb)
        diff_loc = self._local_contrast(diff)
        rgb_edge = self._edge_proxy(rgb)
        diff_edge = self._edge_proxy(diff)

        feat = torch.cat(
            [rgb_avg, rgb_max, rgb_loc, rgb_edge, diff_avg, diff_max, diff_loc, diff_edge], dim=1
        )
        gate = torch.sigmoid(self.net(feat) * self.temp)
        if self.sparse_pow > 1.0:
            gate = gate.pow(self.sparse_pow)
        return self.floor + (1.0 - self.floor) * gate


class PositiveChannelModulator(nn.Module):
    def __init__(self, channel=512, reduction=16, init_bias=-1.8, init_std=1e-3, mod_floor=0.10):
        super().__init__()
        hidden = max(8, channel // max(1, int(reduction)))
        self.pre = Conv(channel * 3, channel, 1, 1)
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.max = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel * 2, hidden, bias=True),
            nn.SiLU(inplace=True),
            nn.Linear(hidden, channel, bias=True),
        )
        nn.init.normal_(self.fc[-1].weight, mean=0.0, std=float(init_std))
        nn.init.constant_(self.fc[-1].bias, float(init_bias))
        self.mod_floor = float(mod_floor)

    def forward(self, rgb, msi_eff):
        diff = torch.abs(rgb - msi_eff)
        z = self.pre(torch.cat([rgb, msi_eff, diff], dim=1))
        z_avg = self.avg(z).flatten(1)
        z_max = self.max(z).flatten(1)
        z = torch.cat([z_avg, z_max], dim=1)
        z = torch.sigmoid(self.fc(z)).unsqueeze(-1).unsqueeze(-1)
        return self.mod_floor + (1.0 - self.mod_floor) * z


class DEA(nn.Module):
    def __init__(
        self,
        channel=512,
        kernel_size=80,
        p_kernel=None,
        m_kernel=None,
        reduction=16,
        g_max=1.0,
        alpha_init=-2.2,
        per_channel=True,
        use_gate=True,
        dropout=0.0,
        beta_init=-2.0,
        bypass_eps_gate=1e-5,
        bypass_eps_msi=1e-8,
        reliability=True,
        rel_reduction=32,
        rel_init=-0.5,
        force_steps=3000,
        force_r=0.65,
        warp=True,
        warp_max_shift=1.5,
        warp_align_corners=False,
        warp_init=0.0,
        warp_learnable=True,
        spatial_select=True,
        sp_reduction=16,
        sp_init=-1.1,
        sp_floor=0.05,
        sp_temp=1.5,
        sp_sparse_pow=2.0,
        mod_init=0.0,
        mod_cap=0.22,
        ch_init_bias=-1.8,
        ch_init_std=1e-3,
        ch_mod_floor=0.10,
    ):
        super().__init__()
        _ = kernel_size, p_kernel, m_kernel

        self.use_gate = bool(use_gate)
        self.g_max = float(g_max)
        self.dropout = float(dropout)
        self.bypass_eps_gate = float(bypass_eps_gate)
        self.bypass_eps_msi = float(bypass_eps_msi)

        self.reliability_on = bool(reliability)
        self.force_steps = int(force_steps)
        self.force_r = float(force_r)
        self.mod_cap = float(mod_cap)

        self.warp_on = bool(warp)
        self.warp = LearnableWarp(max_shift=warp_max_shift, align_corners=warp_align_corners)
        self.sigmoid = nn.Sigmoid()

        if self.use_gate:
            shape = (1, channel, 1, 1) if per_channel else (1,)
            self.alpha = nn.Parameter(torch.full(shape, float(alpha_init)))
            self.beta = nn.Parameter(torch.tensor([float(beta_init)]))

        if self.reliability_on:
            rr = max(8, channel // max(1, int(rel_reduction)))
            self.rel_fuse = Conv(channel * 3, channel, 1, 1)
            self.rel_avg = nn.AdaptiveAvgPool2d(1)
            self.rel_max = nn.AdaptiveMaxPool2d(1)
            self.rel_fc = nn.Sequential(
                nn.Linear(channel * 2, rr, bias=True),
                nn.SiLU(inplace=True),
                nn.Linear(rr, 1, bias=True),
            )
            nn.init.zeros_(self.rel_fc[-1].weight)
            nn.init.constant_(self.rel_fc[-1].bias, float(rel_init))
            self.register_buffer("_step", torch.zeros((), dtype=torch.long), persistent=False)

        if self.warp_on:
            self.warp_learnable = bool(warp_learnable)
            if self.warp_learnable:
                hidden = max(8, channel // 32)
                self.warp_fuse = Conv(channel * 3, channel, 1, 1)
                self.warp_pool = nn.AdaptiveAvgPool2d(1)
                self.warp_fc = nn.Sequential(
                    nn.Linear(channel, hidden, bias=True),
                    nn.SiLU(inplace=True),
                    nn.Linear(hidden, 2, bias=True),
                )
                nn.init.zeros_(self.warp_fc[-1].weight)
                nn.init.constant_(self.warp_fc[-1].bias, float(warp_init))
            else:
                self.register_buffer("warp_dx_const", torch.tensor([float(warp_init)]), persistent=False)
                self.register_buffer("warp_dy_const", torch.tensor([float(warp_init)]), persistent=False)

        self.spatial_select = bool(spatial_select)
        if self.spatial_select:
            self.spatial_gate = LocalModulationGate(
                reduction=sp_reduction,
                init=sp_init,
                floor=sp_floor,
                temp=sp_temp,
                sparse_pow=sp_sparse_pow,
            )

        self.channel_mod = PositiveChannelModulator(
            channel=channel,
            reduction=reduction,
            init_bias=ch_init_bias,
            init_std=ch_init_std,
            mod_floor=ch_mod_floor,
        )

        self.mod_gain = nn.Parameter(torch.tensor([float(mod_init)]))

    def _samplewise_dropout(self, b, device, dtype):
        return (torch.rand(b, 1, 1, 1, device=device) >= self.dropout).to(dtype)

    def _compute_reliability(self, rgb, msi_eff):
        diff = torch.abs(rgb - msi_eff)
        x = self.rel_fuse(torch.cat([rgb, msi_eff, diff], dim=1))
        z_avg = self.rel_avg(x).flatten(1)
        z_max = self.rel_max(x).flatten(1)
        z = torch.cat([z_avg, z_max], dim=1)
        return self.sigmoid(self.rel_fc(z))

    def _compute_warp(self, rgb, msi_eff):
        if self.warp_learnable:
            diff = torch.abs(rgb - msi_eff)
            z = self.warp_fuse(torch.cat([rgb, msi_eff, diff], dim=1))
            z = self.warp_pool(z).flatten(1)
            dxy = self.warp_fc(z)
            dx = dxy[:, 0]
            dy = dxy[:, 1]
        else:
            b = rgb.shape[0]
            dx = self.warp_dx_const.expand(b)
            dy = self.warp_dy_const.expand(b)
        return dx, dy

    def forward(self, x):
        rgb, msi = x[0], x[1]
        b = rgb.shape[0]

        if self.use_gate:
            gate_in = self.sigmoid(self.alpha) * self.g_max
            mix_base = self.sigmoid(self.beta)
            if self.training and self.dropout > 0.0:
                keep_mask = self._samplewise_dropout(b, rgb.device, rgb.dtype)
                gate_in = gate_in * keep_mask
            if not self.training:
                if gate_in.mean().item() < self.bypass_eps_gate:
                    return rgb
                if msi.abs().mean().item() < self.bypass_eps_msi:
                    return rgb
        else:
            gate_in = rgb.new_tensor(1.0)
            mix_base = rgb.new_tensor(1.0)

        msi_eff = msi * gate_in

        if self.warp_on:
            dx, dy = self._compute_warp(rgb, msi_eff)
            msi_eff = self.warp(rgb, msi_eff, dx, dy)

        if self.reliability_on:
            r = self._compute_reliability(rgb, msi_eff)
            if self.training:
                self._step += 1
                if self.force_steps > 0 and int(self._step.item()) < self.force_steps:
                    r_floor = rgb.new_full((b, 1), self.force_r)
                    r = torch.maximum(r, r_floor)
            mix = (mix_base * r).view(b, 1, 1, 1)
        else:
            mix = mix_base.view(1, 1, 1, 1)

        c_mod = self.channel_mod(rgb, msi_eff)
        s_gate = self.spatial_gate(rgb, msi_eff) if self.spatial_select else 1.0

        gain = self.mod_cap * torch.tanh(self.mod_gain).view(1, 1, 1, 1)

        delta = mix * gain * c_mod * s_gate
        out = rgb * (1.0 + delta)
        return out