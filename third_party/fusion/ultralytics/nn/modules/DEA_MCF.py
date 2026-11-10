import torch
import torch.nn as nn
import torch.nn.functional as F
from .conv import Conv

class ZeroConv2d(nn.Conv2d):
    """
    MCF Valve: initialized to zero.
    Ensures Output = RGB + 0 at the start of training.
    """
    def __init__(self, in_ch, out_ch, k=1, s=1, p=0, g=1, bias=True):
        super().__init__(in_ch, out_ch, k, s, p, groups=g, bias=bias)
        # Force weights and bias to zero
        nn.init.zeros_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)


class DEA(nn.Module):
    """
    DEA (DECA + DEPA) with MCF Residual Valve.
    Robust version: Compatible with Ultralytics YAML auto-injected args (c1, c2).
    """
    def __init__(
        self,
        c1=None,          # Auto-passed by YOLO (input channels)
        c2=None,          # Auto-passed by YOLO (output channels)
        channel=512,      # Fallback arg
        kernel_size=80,
        p_kernel=None,
        m_kernel=None,
        reduction=16,
        # --- Legacy args (absorbed to prevent crashes) ---
        g_max=0.30,
        alpha_init=-5.0,
        per_channel=True,
        use_gate=True,
        dropout=0.0,
        beta_init=-5.0,
        *args,
        **kwargs,
    ):
        super().__init__()

        # Robust Channel Resolution
        # YOLO usually passes c1 as input ch, c2 as output ch.
        if c2 is not None:
            ch = int(c2)
        elif c1 is not None:
            ch = int(c1)
        else:
            ch = int(channel)

        # --- A. Attention Modules (DECA + DEPA) ---
        self.deca = DECA(ch, kernel_size, p_kernel, reduction)
        self.depa = DEPA(ch, m_kernel)

        # --- B. Feature Reshaping ---
        # Smooths the distribution of (vi + ir) before the valve
        self.out_conv = Conv(ch, ch, 1, 1)

        # --- C. MCF Valve ---
        # Learns the 'increment' relative to RGB
        self.zero_valve = ZeroConv2d(ch, ch, k=1, s=1, p=0)
        
        # Optional: Keep dropout if you really want it (though usually not needed for MCF)
        self.dropout_p = float(dropout)
        self.drop = nn.Dropout2d(p=self.dropout_p) if self.dropout_p > 0 else nn.Identity()

    def forward(self, x):
        # x[0]: RGB features, x[1]: MSI features
        rgb = x[0]
        
        # 1. Advanced Attention
        deca_out = self.deca(x)
        vi, ir = self.depa(deca_out)
        
        # 2. Fuse & Reshape
        # Simple addition -> Conv smoothing -> Dropout
        fused_raw = vi + ir
        fused_feat = self.out_conv(fused_raw)
        fused_feat = self.drop(fused_feat)
        
        # 3. Calculate Residual Difference
        # The valve learns "what to ADD to RGB"
        delta_in = fused_feat - rgb
        
        # 4. Apply Zero Valve (MCF Core)
        # Init: zero_valve(delta_in) = 0  -> Out = RGB
        out = rgb + self.zero_valve(delta_in)
        
        return out


class DECA(nn.Module):
    """
    Soft-Balanced DPSA (Avg + Max)
    """
    def __init__(self, channel=512, kernel_size=80, p_kernel=None, reduction=16):
        super().__init__()
        self.kernel_size = kernel_size
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
        b, c = pooled_vec.shape
        w = self.fc(pooled_vec).view(b, c, 1, 1)
        return self.act(w)

    def forward(self, x):
        rgb, msi = x[0], x[1]
        b, c, h, w = rgb.shape

        avg_rgb = self.avg_pool(rgb).view(b, c)
        max_rgb = rgb.view(b, c, -1).max(dim=2)[0]
        w_rgb = 0.5 * (self._weight_from_pool(avg_rgb) + self._weight_from_pool(max_rgb))

        avg_msi = self.avg_pool(msi).view(b, c)
        max_msi = msi.view(b, c, -1).max(dim=2)[0]
        w_msi = 0.5 * (self._weight_from_pool(avg_msi) + self._weight_from_pool(max_msi))

        glob_t = self.compress(torch.cat([rgb, msi], dim=1))

        if min(h, w) >= self.kernel_size:
            glob = self.conv_c3(self.conv_c2(self.conv_c1(glob_t)))
            if glob.shape[-2] != 1 or glob.shape[-1] != 1:
                glob = F.adaptive_avg_pool2d(glob, (1, 1))
        else:
            glob = torch.mean(glob_t, dim=[2, 3], keepdim=True)

        rgb_out = rgb * (self.act(w_msi * glob)).expand_as(rgb)
        msi_out = msi * (self.act(w_rgb * glob)).expand_as(msi)

        return [rgb_out, msi_out]


class DEPA(nn.Module):
    """
    Coordinate Spatial Attention (DCSA)
    """
    def __init__(self, channel=512, m_kernel=None, reduction=32):
        super().__init__()
        self.fusion_conv = Conv(channel * 2, channel, 1, 1)

        mip = max(8, channel // reduction)
        self.conv1 = nn.Conv2d(channel, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
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
        y = self.act(self.bn1(self.conv1(y)))

        x_hp, x_wp = torch.split(y, [h, w], dim=2)
        x_wp = x_wp.permute(0, 1, 3, 2)

        a_h = self.sigmoid(self.conv_h(x_hp))
        a_w = self.sigmoid(self.conv_w(x_wp))

        weight = a_h * a_w

        out_rgb = x_rgb * weight
        out_msi = x_msi * weight
        
        return out_rgb, out_msi