import torch
import torch.nn as nn
import torch.nn.functional as F
from .conv import Conv

def global_median_pooling(x):
    """
    Global Median Pooling implementation.
    x: [B, C, H, W] -> [B, C, 1, 1]
    """
    # Flatten H*W dimensions -> [B, C, H*W]
    # Calculate median along the last dimension
    # values are at index [0]
    median_pooled = torch.median(x.view(x.size(0), x.size(1), -1), dim=2)[0]
    return median_pooled.view(x.size(0), x.size(1), 1, 1)

class DEA(nn.Module):
    def __init__(self, channel=512, kernel_size=80, p_kernel=None, m_kernel=None, reduction=16):
        super().__init__()
        # Class name is DECA, but internal logic is updated
        self.deca = DECA(channel, kernel_size, p_kernel, reduction)
        self.depa = DEPA(channel, m_kernel)
        self.act = nn.Sigmoid()

    def forward(self, x):
        # x[0]: RGB, x[1]: MSI
        result_vi, result_ir = self.depa(self.deca(x))
        return self.act(result_vi + result_ir)

class DECA(nn.Module):
    """
    Modified DECA class incorporating MECS-style Channel Attention.
    Logic: Tri-Pooling (Avg + Max + Median)
    """
    def __init__(self, channel=512, kernel_size=80, p_kernel=None, reduction=16):
        super().__init__()
        self.kernel_size = kernel_size
        
        # 1. Shared MLP components
        # Using Conv2d 1x1 as MLP layers (equivalent to Linear but easier for broadcasting)
        internal_neurons = channel // reduction
        self.mlp_fc1 = nn.Conv2d(channel, internal_neurons, kernel_size=1, bias=True)
        self.mlp_relu = nn.ReLU(inplace=True)
        self.mlp_fc2 = nn.Conv2d(internal_neurons, channel, kernel_size=1, bias=True)
        
        # 2. Pooling layers
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # MaxPool and MedianPool are implemented functionally in forward to ensure safety
        
        self.act = nn.Sigmoid()
        self.compress = Conv(channel * 2, channel, 3)

        # 3. Convolution Pyramid for Cross-Modality Interaction
        if p_kernel is None: 
            p_kernel = [5, 4]
        kernel1, kernel2 = p_kernel
        
        self.conv_c1 = nn.Sequential(nn.Conv2d(channel, channel, kernel1, kernel1, 0, groups=channel), nn.SiLU())
        self.conv_c2 = nn.Sequential(nn.Conv2d(channel, channel, kernel2, kernel2, 0, groups=channel), nn.SiLU())
        
        # Dynamic Kernel Calculation for 1280px resolution fix
        k3 = int(self.kernel_size / kernel1 / kernel2)
        self.conv_c3 = nn.Sequential(
            nn.Conv2d(channel, channel, k3, k3, 0, groups=channel),
            nn.SiLU()
        )

    def _get_channel_weight(self, x):
        """
        Calculate Tri-Pooling Channel Attention weights
        """
        # A. Tri-Pooling
        # 1. Average Pooling
        avg_pool = self.avg_pool(x)
        
        # 2. Max Pooling (Functional implementation)
        # Flatten -> Max -> Reshape
        max_pool = x.view(x.size(0), x.size(1), -1).max(dim=2)[0].view(x.size(0), x.size(1), 1, 1)
        
        # 3. Median Pooling (Key Innovation)
        median_pool = global_median_pooling(x)

        # B. Shared MLP Processing
        # Process Avg
        avg_out = self.mlp_fc2(self.mlp_relu(self.mlp_fc1(avg_pool)))
        avg_w = torch.sigmoid(avg_out)
        
        # Process Max
        max_out = self.mlp_fc2(self.mlp_relu(self.mlp_fc1(max_pool)))
        max_w = torch.sigmoid(max_out)
        
        # Process Median
        med_out = self.mlp_fc2(self.mlp_relu(self.mlp_fc1(median_pool)))
        med_w = torch.sigmoid(med_out)
        
        # C. Fusion (Average of 3 branches for stability)
        return (avg_w + max_w + med_w) / 3.0

    def forward(self, x):
        b, c, h, w = x[0].size()
        
        # 1. Apply MECS/Tri-Pooling Channel Attention
        w_vi = self._get_channel_weight(x[0]) # RGB weights
        w_ir = self._get_channel_weight(x[1]) # MSI weights
        
        # 2. Cross-Modality Interaction
        glob_t = self.compress(torch.cat([x[0], x[1]], 1))
        
        if min(h, w) >= self.kernel_size:
            glob = self.conv_c3(self.conv_c2(self.conv_c1(glob_t)))
            # Safety Check for 1280px input (Dynamic Stride Fix)
            if glob.shape[-1] != 1 or glob.shape[-2] != 1:
                glob = F.adaptive_avg_pool2d(glob, (1, 1))
        else:
            glob = torch.mean(glob_t, dim=[2, 3], keepdim=True)

        # 3. Feature Injection
        result_vi = x[0] * (self.act(w_ir * glob)).expand_as(x[0])
        result_ir = x[1] * (self.act(w_vi * glob)).expand_as(x[1])

        return result_vi, result_ir

class DEPA(nn.Module):
    """Standard DEPA module"""
    def __init__(self, channel=512, m_kernel=None):
        super().__init__()
        self.conv1 = Conv(2, 1, 5)
        self.conv2 = Conv(2, 1, 5)
        self.compress1 = Conv(channel, 1, 3)
        self.compress2 = Conv(channel, 1, 3)
        self.act = nn.Sigmoid()

        if m_kernel is None: 
            m_kernel = [3, 7]
        self.cv_v1 = Conv(channel, 1, m_kernel[0])
        self.cv_v2 = Conv(channel, 1, m_kernel[1])
        self.cv_i1 = Conv(channel, 1, m_kernel[0])
        self.cv_i2 = Conv(channel, 1, m_kernel[1])

    def forward(self, x):
        w_vi = self.conv1(torch.cat([self.cv_v1(x[0]), self.cv_v2(x[0])], 1))
        w_ir = self.conv2(torch.cat([self.cv_i1(x[1]), self.cv_i2(x[1])], 1))
        
        glob = self.act(self.compress1(x[0]) + self.compress2(x[1]))
        
        w_vi = self.act(glob + w_vi)
        w_ir = self.act(glob + w_ir)
        
        result_vi = x[0] * w_ir.expand_as(x[0])
        result_ir = x[1] * w_vi.expand_as(x[1])

        return result_vi, result_ir