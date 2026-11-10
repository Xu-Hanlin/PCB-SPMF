import torch
import torch.nn as nn

class SimpleGateFusion(nn.Module):
    """
    Input: [x_rgb, x_msi]
    Output: x_rgb * sigmoid(conv(x_msi))
    """
    def __init__(self, c):
        super().__init__()
        self.gate = nn.Conv2d(c, c, kernel_size=1, stride=1, padding=0, bias=True)
        self.act = nn.Sigmoid()

    def forward(self, x):
        x_rgb, x_msi = x
        g = self.act(self.gate(x_msi))
        return x_rgb * g