import torch
import torch.nn as nn
import torch.nn.functional as F

from cnn import CNN
from mgcn import MGCN
from signal_gcn import GCN
from adaptive_aggregation import AdaptiveAggregation
from lgfm import LGFM
from se_module import SELayer


class DualGraphSHM(nn.Module):
    """Dual-channel Bipartite Feature Graph Network for structural damage detection.

    Args:
        num_sensors: Number of sensor channels (18 for LUMO, 30 for QUGS).
        num_classes: Number of damage classes (7 for LUMO, varies for QUGS).
        adj_norm: Normalized adjacency tensor (without self-loops).
        adj_self: Normalized adjacency tensor (with self-loops).
        feature_dim: CNN output feature dimension (default 200).
        graph_mode: 'dual', 'horizontal', or 'vertical'.
    """

    def __init__(self, num_sensors, num_classes, adj_norm, adj_self,
                 feature_dim=200, graph_mode='dual'):
        super().__init__()
        assert graph_mode in ('dual', 'horizontal', 'vertical')
        self.graph_mode = graph_mode
        self.num_sensors = num_sensors

        # Register adjacency matrices as buffers (move with model.to(device))
        self.register_buffer('adj_norm', adj_norm)
        self.register_buffer('adj_self', adj_self)

        # Shared backbone
        self.cnn = CNN(in_channels=num_sensors, out_channels=num_sensors,
                       feature_dim=feature_dim)

        # Horizontal branch modules
        self.mgcn = MGCN(feature_dim=feature_dim, num_sensors=num_sensors)
        self.signal_gcn = GCN(nfeat=feature_dim, nhid=feature_dim, dropout=0.2)
        self.adaptive = AdaptiveAggregation(
            feature_dim=feature_dim, num_sensors=num_sensors)

        # Vertical branch module
        self.lgfm = LGFM(feature_dim=feature_dim)

        # SE attention for dual mode (created once in __init__)
        h_branch_dim = feature_dim // 2 * 3  # 300 (3 scales x 100 hidden)
        v_branch_dim = 300  # LGFM output
        if graph_mode == 'dual':
            self.se_channel = SELayer(num_sensors)
            self.se_feature = SELayer(v_branch_dim)
            linear_in = num_sensors * (h_branch_dim + v_branch_dim)  # 18*600
        else:
            linear_in = num_sensors * h_branch_dim  # 18*300

        self.classifier = nn.Linear(linear_in, num_classes)

    def forward(self, x):
        x = self.cnn(x)  # (B, num_sensors, feature_dim)

        adj_self = self.adj_self.to(torch.float64)
        adj_norm = self.adj_norm.to(torch.float64)

        B = x.size(0)
        h_branch_dim = 300  # 3 scales x 100 hidden
        v_branch_dim = 300

        G_h = torch.zeros(B, self.num_sensors, h_branch_dim,
                          device=x.device, dtype=torch.float64)
        G_v = torch.zeros(B, self.num_sensors, v_branch_dim,
                          device=x.device, dtype=torch.float64)

        for i in range(B):
            F2 = x[i].to(torch.float64)  # (num_sensors, feature_dim)

            if self.graph_mode in ('dual', 'horizontal'):
                x1, x2, x3, A_F = self.mgcn(F2)
                x_sensor = self.signal_gcn(F2, adj_self)
                H1 = (x1 + x_sensor) / 2
                H2 = (x2 + x_sensor) / 2
                H3 = (x3 + x_sensor) / 2
                A_F = A_F.to(torch.float64).to(x.device)
                H_h = self.adaptive(H1, H2, H3, adj_self, A_F,
                                    x1, x2, x3, x_sensor)
                G_h[i] = H_h

            if self.graph_mode in ('dual', 'vertical'):
                if self.graph_mode == 'vertical':
                    _, _, _, A_F = self.mgcn(F2)
                    A_F = A_F.to(torch.float64).to(x.device)
                H_L = self.lgfm(F2, adj_self, A_F)
                G_v[i] = H_L

        if self.graph_mode == 'horizontal':
            G_output = G_h
        elif self.graph_mode == 'vertical':
            G_output = G_v
        else:
            # SE cross-attention fusion
            dtype = G_h.dtype
            G_h_att = self.se_channel(G_h, G_v)

            G_v_perm = G_v.permute(0, 2, 1)
            G_h_perm = G_h.permute(0, 2, 1)
            G_v_att = self.se_feature(G_v_perm, G_h_perm)
            G_v_att = G_v_att.permute(0, 2, 1)

            G_output = torch.cat([G_h_att, G_v_att], dim=2)

        out = G_output.contiguous().view(B, -1)
        out = self.classifier(out)
        return F.log_softmax(out, dim=1)
