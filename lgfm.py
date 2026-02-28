import torch
import torch.nn as nn

from signal_gcn import GCN


class ForgetGate(nn.Module):
    """LSTM-style forget gate with learnable weights."""

    def __init__(self, hidden_size=50):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(hidden_size * 2, hidden_size))
        self.bias = nn.Parameter(torch.randn(hidden_size))

    def forward(self, ht1, ht2):
        concatenated = torch.cat((ht1, ht2), dim=1)
        a = torch.sigmoid(
            torch.matmul(concatenated.to(self.weight.dtype), self.weight)
            + self.bias)
        return a * ht1


class LGFM(nn.Module):
    """Vertical branch: temporal graph convolution with LSTM-like gating.

    Splits CNN features into 4 time segments (each of size feature_dim/4),
    applies dual GCN paths (feature-graph + spatial-graph) with forget-gate
    connections across time steps.

    Args:
        feature_dim: CNN feature dimension (default 200).
        num_segments: Number of temporal segments (default 4).
        output_dim: Output feature dimension (default 300).
    """

    def __init__(self, feature_dim=200, num_segments=4, output_dim=300):
        super().__init__()
        self.num_segments = num_segments
        seg_size = feature_dim // num_segments  # 50

        self.gcn = GCN(nfeat=seg_size, nhid=seg_size, dropout=0.5)

        # Forget gates for spatial path
        self.gate_s1 = ForgetGate(seg_size)
        self.gate_s2 = ForgetGate(seg_size)
        self.gate_s3 = ForgetGate(seg_size)

        # Forget gates for feature path
        self.gate_f1 = ForgetGate(seg_size)
        self.gate_f2 = ForgetGate(seg_size)
        self.gate_f3 = ForgetGate(seg_size)

        # Forget gates for second-pass spatial path
        self.gate_s1b = ForgetGate(seg_size)
        self.gate_s2b = ForgetGate(seg_size)
        self.gate_s3b = ForgetGate(seg_size)

        # Forget gates for second-pass feature path
        self.gate_f1b = ForgetGate(seg_size)
        self.gate_f2b = ForgetGate(seg_size)
        self.gate_f3b = ForgetGate(seg_size)

        total_concat = num_segments * seg_size * 2  # 4 * 50 * 2 = 400
        self.projection = nn.Linear(total_concat, output_dim)

    def forward(self, x, A_S, A_F):
        seg = x.size(1) // self.num_segments
        x_t1, x_t2, x_t3, x_t4 = (
            x[:, :seg], x[:, seg:2*seg], x[:, 2*seg:3*seg], x[:, 3*seg:])

        # --- Pass 1: Feature-primary path ---
        t1_XF1 = x_t1 + self.gcn(x_t1, A_F)
        t1_XS1 = self.gcn(x_t1, A_S)
        t1_XF2 = 0.2 * t1_XS1 + 0.8 * self.gcn(t1_XF1, A_F)

        t2_XF1 = x_t2 + self.gcn(x_t2, A_F)
        t2_XS1 = self.gcn(x_t2, A_S)
        t2_XS1 = t2_XS1 + self.gate_s1(t1_XS1, t2_XS1)
        t2_XF2 = 0.2 * t2_XS1 + 0.8 * self.gcn(t2_XF1, A_F)
        t2_XF2 = self.gate_f1(t1_XF2, t2_XF2) + t2_XF2

        t3_XF1 = 0.2 * x_t3 + 0.8 * self.gcn(x_t3, A_F)
        t3_XS1 = self.gcn(x_t3, A_S)
        t3_XS1 = t3_XS1 + self.gate_s2(t2_XS1, t3_XS1)
        t3_XF2 = 0.2 * t3_XS1 + 0.8 * self.gcn(t3_XF1, A_F)
        t3_XF2 = self.gate_f2(t2_XF2, t3_XF2) + t3_XF2

        t4_XF1 = x_t4 + self.gcn(x_t4, A_F)
        t4_XS1 = self.gcn(x_t4, A_S)
        t4_XS1 = t4_XS1 + self.gate_s3(t3_XS1, t4_XS1)
        t4_XF2 = 0.2 * t4_XS1 + 0.8 * self.gcn(t4_XF1, A_F)
        t4_XF2 = self.gate_f3(t3_XF2, t4_XF2) + t4_XF2

        # --- Pass 2: Spatial-primary path ---
        t1_XS1b = x_t1 + self.gcn(x_t1, A_S)
        t1_XF1b = self.gcn(x_t1, A_F)
        t1_XS2 = 0.2 * t1_XF1b + 0.8 * self.gcn(t1_XS1b, A_S)

        t2_XS1b = x_t2 + self.gcn(x_t2, A_S)
        t2_XF1b = self.gcn(x_t2, A_F)
        t2_XF1b = t2_XF1b + self.gate_f1b(t1_XF1b, t2_XF1b)
        t2_XS2 = 0.2 * t2_XF1b + 0.8 * self.gcn(t2_XS1b, A_S)
        t2_XS2 = self.gate_s1b(t1_XS2, t2_XS2) + t2_XS2

        t3_XS1b = 0.2 * x_t3 + 0.8 * self.gcn(x_t3, A_S)
        t3_XF1b = self.gcn(x_t3, A_F)
        t3_XF1b = t3_XF1b + self.gate_f2b(t2_XF1b, t3_XF1b)
        t3_XS2 = 0.2 * t3_XF1b + 0.8 * self.gcn(t3_XS1b, A_S)
        t3_XS2 = self.gate_s2b(t2_XS2, t3_XS2) + t3_XS2

        t4_XS1b = x_t4 + self.gcn(x_t4, A_S)
        t4_XF1b = self.gcn(x_t4, A_F)
        t4_XF1b = t4_XF1b + self.gate_f3b(t3_XF1b, t4_XF1b)
        t4_XS2 = 0.2 * t4_XF1b + 0.8 * self.gcn(t4_XS1b, A_S)
        t4_XS2 = self.gate_s3b(t3_XS2, t4_XS2) + t4_XS2

        # Concatenate both paths across all time steps
        h1 = torch.cat([t1_XF2, t1_XS2], dim=1)
        h2 = torch.cat([t2_XF2, t2_XS2], dim=1)
        h3 = torch.cat([t3_XF2, t3_XS2], dim=1)
        h4 = torch.cat([t4_XF2, t4_XS2], dim=1)

        h_L = torch.cat([h1, h2, h3, h4], dim=1)
        return self.projection(h_L)
