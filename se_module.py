import torch
import torch.nn as nn


class SELayer(nn.Module):
    """Dual-input Squeeze-and-Excitation attention layer.

    Computes channel attention from two input branches and applies
    the fused attention weights to the first branch.
    """

    def __init__(self, channel, reduction=8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, x1):
        b, c, _ = x.size()
        b1, c1, _ = x1.size()

        y = self.avg_pool(x).view(b, c)
        y1 = self.avg_pool(x1).view(b1, c1)

        y = self.fc(y).view(b, c, 1)
        y1 = self.fc(y1).view(b1, c1, 1)

        y = self.sigmoid(y + y1)
        return x * y.expand_as(x)
