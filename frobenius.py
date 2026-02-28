import torch
import torch.nn as nn


class FrobeniusFusion(nn.Module):
    """Frobenius-norm-weighted fusion of spatial and feature adjacency matrices."""

    def __init__(self, feature_dim=200, num_sensors=18):
        super().__init__()
        self.num_sensors = num_sensors
        self.layer1 = nn.Sequential(
            nn.Linear(feature_dim, 1),
            nn.Sigmoid(),
        )
        self.layer2 = nn.Sequential(
            nn.Linear(feature_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, A_S, A_F, A1, X_sensor, H):
        D_diff1 = A1 - A_S
        D_diff2 = A1 - A_F

        fnorm1 = torch.norm(D_diff1, p='fro')
        fnorm2 = torch.norm(D_diff2, p='fro')

        a1 = fnorm1 / (fnorm1 + fnorm2)
        a2 = fnorm2 / (fnorm1 + fnorm2)

        eye = torch.eye(self.num_sensors, device=A_S.device)
        A_S = A_S * eye
        A_F = A_F * eye

        if fnorm1 > fnorm2:
            aF = self.layer1(H)
            A = aF * (a2 * A_S + a1 * A_F)
        else:
            aS = self.layer2(X_sensor)
            A = aS * (a2 * A_S + a1 * A_F)

        return A
