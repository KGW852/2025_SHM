# models/modules/simsiam.py

import torch.nn as nn

class SimSiamProjector(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),

            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),

            nn.Linear(hidden_dim, out_dim),
            nn.BatchNorm1d(out_dim)
        )

    def forward(self, x):  # x: (B, in_dim)
        return self.mlp(x)
    
class SimSiamPredictor(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),

            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, x):  # x: (B, in_dim)
        return self.mlp(x)

class SimSiam(nn.Module):
    def __init__(self, in_dim, proj_hidden_dim, proj_out_dim, pred_hidden_dim, pred_out_dim):
        super(SimSiam, self).__init__()
        self.projector = SimSiamProjector(
            in_dim=in_dim,
            hidden_dim=proj_hidden_dim,
            out_dim=proj_out_dim
        )
        self.predictor = SimSiamPredictor(
            in_dim=proj_out_dim,
            hidden_dim=pred_hidden_dim,
            out_dim=pred_out_dim
        )

    def forward(self, x1, x2):
        # projection
        z1 = self.projector(x1)  # (B, proj_out_dim)
        z2 = self.projector(x2)  # (B, proj_out_dim)

        # prediction
        p1 = self.predictor(z1)  # (B, pred_out_dim)
        p2 = self.predictor(z2)  # (B, pred_out_dim)

        return z1, p1, z2, p2