import torch.nn as nn


class CNN(nn.Module):
    """4-layer 1D-CNN backbone for extracting features from raw sensor signals.

    Args:
        in_channels: Number of input sensor channels (18 for LUMO, 30 for QUGS).
        out_channels: Number of output channels (typically equals in_channels).
        feature_dim: Adaptive pooling target length (default 200).
    """

    def __init__(self, in_channels=18, out_channels=18, feature_dim=200):
        super().__init__()

        self.layer1 = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=15, padding=7),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
        )
        self.layer2 = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )
        self.layer3 = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
        )
        self.layer4 = nn.Sequential(
            nn.Conv1d(64, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.AdaptiveMaxPool1d(feature_dim),
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x
