# models/convergent_sim.py

import torch
import torch.nn as nn
from .modules.encoder import MLPEncoder
from .modules.simsiam import SimSiam

class SimEncoder(nn.Module):
    """
    Combine MLPEncoder and SimSiam for source/target domain inputs.
    Forward:
        x_s, x_t: (B, C, T)
        Returns:
            e_s, e_t: encoder outputs (latent_dim).
            z_s, p_s, z_t, p_t: SimSiam projector/predictor results.
    """
    def __init__(self,
                 in_dim: int,
                 hidden_dims: list,
                 latent_dim: int,
                 dropout: float = 0.0,
                 proj_hidden_dim: int = 512,
                 proj_out_dim: int = 512,
                 pred_hidden_dim: int = 256,
                 pred_out_dim: int = 512):
        super(SimEncoder, self).__init__()

        # MLP Encoder
        self.encoder = MLPEncoder(in_dim=in_dim,
                                  hidden_dims=hidden_dims,
                                  latent_dim=latent_dim,
                                  dropout=dropout)
        
        # SimSiam
        self.simsiam = SimSiam(
            in_dim=latent_dim,
            proj_hidden_dim=proj_hidden_dim,
            proj_out_dim=proj_out_dim,
            pred_hidden_dim=pred_hidden_dim,
            pred_out_dim=pred_out_dim
        )

    def forward(self, x_s: torch.Tensor, x_t: torch.Tensor):
        e_s = self.encoder(x_s)  # (B, latent_dim)
        e_t = self.encoder(x_t)

        z_s, p_s, z_t, p_t = self.simsiam(e_s, e_t)

        return e_s, e_t, z_s, p_s, z_t, p_t