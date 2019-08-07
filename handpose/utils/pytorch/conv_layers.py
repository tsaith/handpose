import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

class Conv1dSame(torch.nn.Module):
    """
    Conv1d with fixed shape. Assumed stride=1, dilation=1, groups=1.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1,
                 bias=True, groups=1, padding_layer=nn.ConstantPad1d):
        super().__init__()

        ka = kernel_size // 2
        kb = ka - 1 if kernel_size % 2 == 0 else ka
        self.net = nn.Sequential(
            padding_layer((ka, kb), 0.0),
            nn.Conv1d(in_channels, out_channels, kernel_size, bias=bias, groups=groups)
        )

    def forward(self, x):
        return self.net(x)


class Conv2dSame(torch.nn.Module):
    """
    Conv2d with fixed shape. Assumed stride=1, dilation=1, groups=1.
    """
    def __init__(self, in_channels, out_channels, kernel_size,
                 bias=True, groups=1, padding_layer=torch.nn.ConstantPad2d):
        super().__init__()
        ka = kernel_size // 2
        kb = ka - 1 if kernel_size % 2 == 0 else ka
        self.net = nn.Sequential(
            padding_layer((ka,kb,ka,kb), 0.0),
            nn.Conv2d(in_channels, out_channels, kernel_size, bias=bias, groups=groups)
        )

    def forward(self, x):
        return self.net(x)


