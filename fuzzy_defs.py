# fuzzy_defs.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class GaussMF(nn.Module):
    def __init__(self, mean: float, sigma: float):
        super().__init__()
        self.mean = nn.Parameter(torch.tensor(mean, dtype=torch.float32))
        self.sigma = nn.Parameter(torch.tensor(sigma, dtype=torch.float32))

    def forward(self, x):
        # x shape: (N,)
        return torch.exp(-0.5 * ((x - self.mean) / (self.sigma + 1e-8)) ** 2)

class ANFISLayer(nn.Module):
    def __init__(self, centers_x, centers_y, sigma_x=2.0, sigma_y=2.0):
        super().__init__()
        self.mfs_x = nn.ModuleList([GaussMF(c, sigma_x) for c in centers_x])
        self.mfs_y = nn.ModuleList([GaussMF(c, sigma_y) for c in centers_y])

    def forward(self, x, y):
        # x,y shape (N,)
        mux = torch.stack([mf(x) for mf in self.mfs_x], dim=1)  # (N, Mx)
        muy = torch.stack([mf(y) for mf in self.mfs_y], dim=1)  # (N, My)
        rules = mux.unsqueeze(2) * muy.unsqueeze(1)  # (N, Mx, My)
        rules = rules.view(x.size(0), -1)  # (N, Mx*My)
        return rules

class AnfisNet(nn.Module):
    def __init__(self, centers_x=(5.0, 9.5, 14.0), centers_y=(5.0, 9.5, 14.0), sigma=3.0):
        super().__init__()
        self.layer = ANFISLayer(centers_x, centers_y, sigma_x=sigma, sigma_y=sigma)
        # consequent linear layer mapping normalized rule strengths -> output angle
        self.consequent = nn.Linear(9, 1)

    def forward(self, X):
        # X: (N,2)
        x = X[:, 0]
        y = X[:, 1]
        rules = self.layer(x, y)  # (N,9)
        weights = F.softmax(rules, dim=1)
        out = self.consequent(weights)
        return out
