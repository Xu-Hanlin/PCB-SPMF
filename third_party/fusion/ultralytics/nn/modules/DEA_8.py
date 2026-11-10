import torch
import torch.nn as nn

class DEA(nn.Module):
    def __init__(self, c, *args, **kwargs):
        super().__init__()
        self.c = c
        self.proj = nn.Conv2d(c, c, 1, 1, 0, bias=True)
        self.gate = nn.Parameter(torch.zeros(1, c, 1, 1))
        self.act = nn.SiLU()
        self.reset_parameters()

        # keep your old submodules if you want
        # self.xxx = ...

    def reset_parameters(self):
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)
        with torch.no_grad():
            self.gate.zero_()

    def forward(self, x):
        rgb, msi = x[0], x[1]
        r = self.proj(msi)
        g = torch.tanh(self.gate)
        out = rgb + g * self.act(r)
        return out
