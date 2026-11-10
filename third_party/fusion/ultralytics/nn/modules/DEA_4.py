"""
DEA (DECA and DEPA) module - Final Ultimate Version
1. DECA: Soft-Balanced DPSA (Avg + Max) for High-Frequency Feature Extraction.
2. DEPA: Upgraded to Coordinate Spatial Attention (DCSA) for Elongated Object Localization.
3. 1280px High-Res Support Included.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .conv import Conv

class DEA(nn.Module):
    """x0 --> RGB feature map,  x1 --> IR/MSI feature map"""
    def __init__(self, channel=512, kernel_size=80, p_kernel=None, m_kernel=None, reduction=16):
        super().__init__()
        self.deca = DECA(channel, kernel_size, p_kernel, reduction)
        self.depa = DEPA(channel, m_kernel) # Now upgrades to Coordinate Attention internally
        self.act = nn.Sigmoid()

    def forward(self, x):
        # x[0]: RGB, x[1]: MSI
        # Step 1: Channel Attention (DECA/DPSA)
        deca_out = self.deca(x)
        # Step 2: Spatial Attention (DEPA/Coordinate)
        result_vi, result_ir = self.depa(deca_out)
        return self.act(result_vi + result_ir)


class DECA(nn.Module):
    """
    [Channel Attention]
    Soft-Balanced DPSA (Dual-Pooling Spectral Attention).
    Strategy: AvgPool (Background) + MaxPool (Texture/Highlight).
    Weight = (Sigmoid(MLP(Avg)) + Sigmoid(MLP(Max))) / 2
    """
    def __init__(self, channel=512, kernel_size=80, p_kernel=None, reduction=16):
        super().__init__()
        self.kernel_size = kernel_size
        
        # 1. Pooling Layers
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # MaxPool is implemented functionally in forward
        
        # 2. Shared MLP (No Sigmoid at the end!)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False)
        )
        
        self.act = nn.Sigmoid()
        self.compress = Conv(channel * 2, channel, 3)

        # 3. Convolution Pyramid
        if p_kernel is None: p_kernel = [5, 4]
        kernel1, kernel2 = p_kernel
        self.conv_c1 = nn.Sequential(nn.Conv2d(channel, channel, kernel1, kernel1, 0, groups=channel), nn.SiLU())
        self.conv_c2 = nn.Sequential(nn.Conv2d(channel, channel, kernel2, kernel2, 0, groups=channel), nn.SiLU())
        k3 = int(self.kernel_size / kernel1 / kernel2)
        self.conv_c3 = nn.Sequential(nn.Conv2d(channel, channel, k3, k3, 0, groups=channel), nn.SiLU())

    def _get_weight(self, feat):
        return self.act(self.fc(feat))

    def forward(self, x):
        b, c, h, w = x[0].size()
        
        # === Soft-Balanced Fusion (Avg + Max) ===
        # RGB Branch
        avg_vi = self.avg_pool(x[0]).view(b, c)
        max_vi = x[0].view(b, c, -1).max(dim=2)[0] 
        w_vi = (self._get_weight(avg_vi) + self._get_weight(max_vi)) * 0.5
        w_vi = w_vi.view(b, c, 1, 1)

        # MSI Branch
        avg_ir = self.avg_pool(x[1]).view(b, c)
        max_ir = x[1].view(b, c, -1).max(dim=2)[0]
        w_ir = (self._get_weight(avg_ir) + self._get_weight(max_ir)) * 0.5
        w_ir = w_ir.view(b, c, 1, 1)
        
        # Cross-Modal Interaction (With 1280px fix)
        glob_t = self.compress(torch.cat([x[0], x[1]], 1))
        if min(h, w) >= self.kernel_size:
            glob = self.conv_c3(self.conv_c2(self.conv_c1(glob_t)))
            if glob.shape[-1] != 1 or glob.shape[-2] != 1:
                glob = F.adaptive_avg_pool2d(glob, (1, 1))
        else:
            glob = torch.mean(glob_t, dim=[2, 3], keepdim=True)

        result_vi = x[0] * (self.act(w_ir * glob)).expand_as(x[0])
        result_ir = x[1] * (self.act(w_vi * glob)).expand_as(x[1])

        return [result_vi, result_ir] # Return as list for DEPA input


class DEPA(nn.Module):
    """
    [Spatial Attention]
    Upgraded to: Dual Coordinate Spatial Attention (DCSA)
    Why? Standard 7x7 conv is bad for long/thin connectors.
    Coordinate Attention pools along X and Y axes separately to capture long-range dependencies.
    """
    def __init__(self, channel=512, m_kernel=None):
        super().__init__()
        
        # Coordinate Attention parameters
        reduction = 32
        mip = max(8, channel // reduction)
        
        # 1. Early Fusion of RGB+MSI
        self.fusion_conv = Conv(channel * 2, channel, 1)
        
        # 2. Shared 1x1 Conv
        self.conv1 = nn.Conv2d(channel, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = nn.SiLU() # Modern activation
        
        # 3. Split Convs for H and W
        self.conv_h = nn.Conv2d(mip, channel, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, channel, kernel_size=1, stride=1, padding=0)
        
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x comes from DECA output: [result_vi, result_ir]
        x_rgb, x_msi = x[0], x[1]
        b, c, h, w = x_rgb.size()
        
        # === Step 1: Early Fusion & Coordinate Pooling ===
        # Merge features to find spatial focus shared by both modalities
        x_fuse = self.fusion_conv(torch.cat([x_rgb, x_msi], dim=1))
        
        # Pool X (H direction) -> (b, c, h, 1)
        x_h = x_fuse.mean(dim=3, keepdim=True)
        # Pool Y (W direction) -> (b, c, 1, w) -> Permute to (b, c, w, 1) for stacking
        x_w = x_fuse.mean(dim=2, keepdim=True).permute(0, 1, 3, 2)
        
        # === Step 2: Coordinate Attention Generation ===
        # Concatenate along spatial dimension for shared processing
        y = torch.cat([x_h, x_w], dim=2) # (b, c, h+w, 1)
        
        # Compress
        y = self.act(self.bn1(self.conv1(y)))
        
        # Split back
        x_h_prime, x_w_prime = torch.split(y, [h, w], dim=2)
        
        # Permute W back
        x_w_prime = x_w_prime.permute(0, 1, 3, 2)
        
        # Expand
        a_h = self.sigmoid(self.conv_h(x_h_prime)) # Attention Map H
        a_w = self.sigmoid(self.conv_w(x_w_prime)) # Attention Map W
        
        # === Step 3: Apply Attention ===
        # Final Spatial Weight = Attn_H * Attn_W
        spatial_weight = a_h * a_w
        
        # Dual Injection
        result_vi = x_rgb * spatial_weight
        result_ir = x_msi * spatial_weight
        
        return result_vi, result_ir