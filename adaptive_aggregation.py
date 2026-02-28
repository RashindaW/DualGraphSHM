import torch
import torch.nn as nn

from signal_gcn import GCN
from frobenius import FrobeniusFusion


class AdaptiveAggregation(nn.Module):
    """Adaptive adjacency fusion for the horizontal branch.

    Fuses spatial and feature-learned adjacency matrices using Frobenius
    norm weighting, then applies GCN at each Chebyshev scale.

    Args:
        feature_dim: Feature dimension from CNN (default 200).
        num_sensors: Number of sensor nodes (18 for LUMO, 30 for QUGS).
    """

    def __init__(self, feature_dim=200, num_sensors=18):
        super().__init__()
        self.layer1 = nn.Sequential(nn.Linear(feature_dim, 1), nn.Sigmoid())
        self.layer2 = nn.Sequential(nn.Linear(feature_dim, 1), nn.Sigmoid())
        self.layer3 = nn.Sequential(nn.Linear(feature_dim, 1), nn.Sigmoid())

        gcn_hid = feature_dim // 2
        self.gcn1 = GCN(nfeat=feature_dim, nhid=gcn_hid, dropout=0.5)
        self.gcn2 = GCN(nfeat=feature_dim, nhid=gcn_hid, dropout=0.5)
        self.gcn3 = GCN(nfeat=feature_dim, nhid=gcn_hid, dropout=0.5)

        self.frobenius = FrobeniusFusion(feature_dim, num_sensors)

    def forward(self, H1, H2, H3, A_S, A_F, x1, x2, x3, x_sensor):
        A_S = A_S.to_dense()

        a1 = self.layer1(H1)
        A1 = a1 * A_S + (1 - a1) * A_F
        A1 = A1 + self.frobenius(A_S, A_F, A1, x_sensor, x1)
        H1 = self.gcn1(H1, A1)

        a2 = self.layer2(H2)
        A2 = a2 * A_S + (1 - a2) * A_F
        A2 = A2 + self.frobenius(A_S, A_F, A2, x_sensor, x2)
        H2 = self.gcn2(H2, A2)

        a3 = self.layer3(H3)
        A3 = a3 * A_S + (1 - a3) * A_F
        A3 = A3 + self.frobenius(A_S, A_F, A3, x_sensor, x3)
        H3 = self.gcn3(H3, A3)

        return torch.cat([H1, H2, H3], dim=1)
