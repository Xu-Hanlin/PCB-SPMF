import torch.nn as nn

class PassThrough(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return x

class SelectInput(nn.Module):
    def __init__(self, index=0):
        super().__init__()
        self.index = index
    def forward(self, x):
        if isinstance(x, (list, tuple)):
            return x[self.index]
        return x
