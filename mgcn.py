import torch
import torch.nn as nn
from torch.nn.functional import cosine_similarity
from torch_geometric.nn import ChebConv, BatchNorm
from torch_geometric.utils import dropout_adj


class GGL(nn.Module):
    """Graph Generation Layer: learns adjacency via cosine similarity."""

    def __init__(self, feature_dim=200, num_sensors=18):
        super().__init__()
        self.num_sensors = num_sensors
        self.layer = nn.Sequential(
            nn.Linear(feature_dim, num_sensors),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        attr = self.layer(x)
        values, edge_index, A_norm = self._gen_edge(attr)
        return values.view(-1), edge_index, A_norm

    def _gen_edge(self, attr):
        n = self.num_sensors
        attr_cpu = attr.cpu()
        adj = torch.zeros(n, n)
        for i in range(n):
            for j in range(n):
                adj[i, j] = cosine_similarity(
                    attr_cpu[i].unsqueeze(0), attr_cpu[j].unsqueeze(0))

        maxval, _ = adj.max(dim=1)
        A_norm = adj / maxval.unsqueeze(1)
        k = n
        values, indices = A_norm.topk(k, dim=1, largest=True, sorted=False)

        edge_index = torch.tensor([[], []], dtype=torch.long)
        for i in range(n):
            src = torch.zeros(indices.shape[1], dtype=torch.long) + i
            edge_index = torch.cat(
                [edge_index, torch.stack([src, indices[i]])], dim=1)

        return values, edge_index, A_norm


class MultiChev1(nn.Module):
    """Multi-scale Chebyshev convolution (K=1,2,3)."""

    def __init__(self, in_channels, out_channels=200):
        super().__init__()
        self.scale_1 = ChebConv(in_channels, out_channels, K=1)
        self.scale_2 = ChebConv(in_channels, out_channels, K=2)
        self.scale_3 = ChebConv(in_channels, out_channels, K=3)

    def forward(self, x, edge_index, edge_weight):
        s1 = self.scale_1(x, edge_index, edge_weight)
        s2 = self.scale_2(x, edge_index, edge_weight)
        s3 = self.scale_3(x, edge_index, edge_weight)
        return s1, s2, s3


class MultiChevB(nn.Module):
    """Multi-scale Chebyshev convolution (K=2,3,4), concatenated output."""

    def __init__(self, in_channels, out_channels=100):
        super().__init__()
        self.scale_1 = ChebConv(in_channels, out_channels, K=2)
        self.scale_2 = ChebConv(in_channels, out_channels, K=3)
        self.scale_3 = ChebConv(in_channels, out_channels, K=4)

    def forward(self, x, edge_index, edge_weight):
        s1 = self.scale_1(x, edge_index, edge_weight)
        s2 = self.scale_2(x, edge_index, edge_weight)
        s3 = self.scale_3(x, edge_index, edge_weight)
        return torch.cat([s1, s2, s3], dim=1)


class MGCN(nn.Module):
    """Multi-scale GCN: graph generation + multi-scale Chebyshev convolution.

    Args:
        feature_dim: CNN feature dimension (default 200).
        num_sensors: Number of sensor nodes (18 for LUMO, 30 for QUGS).
    """

    def __init__(self, feature_dim=200, num_sensors=18):
        super().__init__()
        self.ggl = GGL(feature_dim, num_sensors)
        self.conv1 = MultiChev1(feature_dim)
        self.bn1 = BatchNorm(200)

    def forward(self, x):
        device = x.device
        edge_attr, edge_index, A_norm = self.ggl(x)
        edge_attr = edge_attr.to(device)
        edge_index = edge_index.to(device)
        edge_index, edge_attr = dropout_adj(edge_index, edge_attr)

        x1, x2, x3 = self.conv1(x, edge_index, edge_weight=edge_attr)
        x1 = self.bn1(x1)
        x2 = self.bn1(x2)
        x3 = self.bn1(x3)
        return x1, x2, x3, A_norm
