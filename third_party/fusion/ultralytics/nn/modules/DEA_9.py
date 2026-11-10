import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.nn.modules.conv import Conv


def _make_base_grid(b, h, w, device, dtype):
    ys = torch.linspace(-1.0, 1.0, h, device=device, dtype=dtype)
    xs = torch.linspace(-1.0, 1.0, w, device=device, dtype=dtype)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    grid = torch.stack([grid_x, grid_y], dim=-1)  # (H,W,2)
    return grid.unsqueeze(0).repeat(b, 1, 1, 1)  # (B,H,W,2)


class FeatureAlign(nn.Module):
    def __init__(
        self,
        channel,
        max_disp=4.0,
        align_mode="warp",
        align_level="global",
        align_kernel=3,
        init_zero=True,
    ):
        super().__init__()
        self.max_disp = float(max_disp)
        self.align_mode = str(align_mode)
        self.align_level = str(align_level)
        k = int(align_kernel)

        if self.align_mode not in ("warp",):
            raise ValueError("align_mode must be 'warp'")

        if self.align_level not in ("global", "pixel"):
            raise ValueError("align_level must be 'global' or 'pixel'")

        if self.align_level == "global":
            self.net = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(channel * 2, max(16, channel // 16), 1, 1, 0, bias=True),
                nn.SiLU(inplace=True),
                nn.Conv2d(max(16, channel // 16), 2, 1, 1, 0, bias=True),
            )
        else:
            pad = k // 2
            self.net = nn.Sequential(
                nn.Conv2d(channel * 2, max(32, channel // 8), k, 1, pad, bias=True),
                nn.SiLU(inplace=True),
                nn.Conv2d(max(32, channel // 8), 2, 1, 1, 0, bias=True),
            )

        if init_zero:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.zeros_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

    def forward(self, rgb, msi):
        b, c, h, w = rgb.shape
        x = torch.cat([rgb, msi], dim=1)
        flow = self.net(x)  # (B,2,1,1) or (B,2,H,W)

        if flow.shape[-2:] == (1, 1):
            flow = flow.expand(b, 2, h, w)

        flow = torch.tanh(flow) * self.max_disp  # pixels

        dx = flow[:, 0, :, :]  # (B,H,W)
        dy = flow[:, 1, :, :]  # (B,H,W)

        dx_norm = dx / max(1.0, (w - 1)) * 2.0
        dy_norm = dy / max(1.0, (h - 1)) * 2.0

        grid = _make_base_grid(b, h, w, rgb.device, rgb.dtype)
        grid[..., 0] = grid[..., 0] + dx_norm
        grid[..., 1] = grid[..., 1] + dy_norm

        msi_warp = F.grid_sample(
            msi,
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )
        return msi_warp, flow


class DEA(nn.Module):
    """
    YAML args order:
    [channel, kernel_size, p_kernel, m_kernel, reduction, g_max, alpha_init, per_channel, use_gate]

    Extra optional args:
    - dropout: sample-wise modality dropout probability
    - beta_init: init value for mix gate
    - bypass_eps_gate: inference bypass threshold on gate mean
    - bypass_eps_msi: inference bypass threshold on MSI magnitude
    - use_align: enable MSI alignment before fusion
    - align_level: "global" or "pixel"
    - align_max_disp: max displacement in pixels
    - align_kernel: kernel size for pixel-level align net
    - align_init_zero: init align net to identity
    - align_gate_min: only align when gate mean >= this
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
        use_align=True,
        align_level="global",
        align_max_disp=4.0,
        align_kernel=3,
        align_init_zero=True,
        align_gate_min=1e-3,
    ):
        super().__init__()
        self.deca = DECA(channel, kernel_size, p_kernel, reduction)
        self.depa = DEPA(channel, m_kernel)

        self.use_gate = bool(use_gate)
        self.g_max = float(g_max)
        self.dropout = float(dropout)
        self.bypass_eps_gate = float(bypass_eps_gate)
        self.bypass_eps_msi = float(bypass_eps_msi)

        self.use_align = bool(use_align)
        self.align_gate_min = float(align_gate_min)
        if self.use_align:
            self.align = FeatureAlign(
                channel=channel,
                max_disp=align_max_disp,
                align_mode="warp",
                align_level=align_level,
                align_kernel=align_kernel,
                init_zero=align_init_zero,
            )

        if self.use_gate:
            shape = (1, channel, 1, 1) if per_channel else (1,)
            self.alpha = nn.Parameter(torch.full(shape, float(alpha_init)))
            self.beta = nn.Parameter(torch.tensor([float(beta_init)]))
            self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        rgb, msi = x[0], x[1]
        b = rgb.shape[0]

        if self.use_gate:
            gate_in = self.sigmoid(self.alpha) * self.g_max
            mix_weight = self.sigmoid(self.beta)

            if self.training and self.dropout > 0.0:
                keep_mask = (torch.rand(b, 1, 1, 1, device=rgb.device) >= self.dropout).to(rgb.dtype)
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

        if self.use_align:
            do_align = True
            if gate_in.mean().item() < self.align_gate_min:
                do_align = False
            if not self.training and msi_eff.abs().mean().item() < self.bypass_eps_msi:
                do_align = False

            if do_align:
                msi_eff, _ = self.align(rgb, msi_eff)

        deca_out = self.deca([rgb, msi_eff])
        result_vi, result_ir = self.depa(deca_out)

        fused_rgb = result_vi + result_ir
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
    def __init__(self, channel=512, m_kernel=None):
        super().__init__()
        reduction = 32
        mip = max(8, channel // reduction)

        self.fusion_conv = Conv(channel * 2, channel, 1)

        self.conv1 = nn.Conv2d(channel, mip, kernel_size=1, stride=1, padding=0)

        for g in (32, 16, 8, 4, 2, 1):
            if mip % g == 0:
                num_groups = g
                break
        self.gn1 = nn.GroupNorm(num_groups=num_groups, num_channels=mip)

        self.act = nn.SiLU(inplace=True)
        self.conv_h = nn.Conv2d(mip, channel, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, channel, kernel_size=1, stride=1, padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x_rgb, x_msi = x[0], x[1]
        b, c, h, w = x_rgb.size()

        x_fuse = self.fusion_conv(torch.cat([x_rgb, x_msi], dim=1))

        x_h = x_fuse.mean(dim=3, keepdim=True)
        x_w = x_fuse.mean(dim=2, keepdim=True).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.act(self.gn1(self.conv1(y)))

        x_h_prime, x_w_prime = torch.split(y, [h, w], dim=2)
        x_w_prime = x_w_prime.permute(0, 1, 3, 2)

        a_h = self.sigmoid(self.conv_h(x_h_prime))
        a_w = self.sigmoid(self.conv_w(x_w_prime))

        spatial_weight = a_h * a_w

        result_vi = x_rgb * spatial_weight
        result_ir = x_msi * spatial_weight
        return result_vi, result_ir
