# models/convergent_sim.py

import torch
import torch.nn as nn
from .modules.encoder import MLPEncoder
from .modules.simsiam import SimSiam
from .modules.deep_svdd import DeepSVDD

class ConvergentSim(nn.Module):
    """
    Combine MLPEncoder and SimSiam for source/target domain inputs.
    then pass z_s, z_t to DeepSVDD for additional feature processing (or identity).
    Forward:
        x_s, x_t: (B, C, T)
    Returns:
        e_s, e_t: encoder outputs (latent_dim).
        z_s, p_s, z_t, p_t: SimSiam projector/predictor results.
        svdd_feat_s, svdd_feat_t : DeepSVDD feature outputs (from z_s, z_t)
    """
    def __init__(self,
                 enc_in_dim: int,
                 enc_hidden_dims: list,
                 enc_latent_dim: int,
                 dropout: float = 0.0,
                 proj_hidden_dim: int = 64,
                 proj_out_dim: int = 64,
                 pred_hidden_dim: int = 32,
                 pred_out_dim: int = 64,
                 svdd_latent_dim: int = 64):
        super(ConvergentSim, self).__init__()

        # MLP Encoder
        self.encoder = MLPEncoder(
            in_dim=enc_in_dim,
            hidden_dims=enc_hidden_dims,
            latent_dim=enc_latent_dim,
            dropout=dropout)
        
        # SimSiam
        self.simsiam = SimSiam(
            in_dim=enc_latent_dim,
            proj_hidden_dim=proj_hidden_dim,
            proj_out_dim=proj_out_dim,
            pred_hidden_dim=pred_hidden_dim,
            pred_out_dim=pred_out_dim)

        # DeepSVDD
        self.deep_svdd = DeepSVDD(
            backbone=nn.Identity(), 
            latent_dim=svdd_latent_dim)

    def forward(self, x_s: torch.Tensor, x_t: torch.Tensor):
        e_s = self.encoder(x_s)  # (B, latent_dim)
        e_t = self.encoder(x_t)

        z_s, p_s, z_t, p_t = self.simsiam(e_s, e_t)

        svdd_feat_s = self.deep_svdd(z_s)  # (B, svdd_latent_dim)
        svdd_feat_t = self.deep_svdd(z_t)

        return (
            e_s, e_t,       # encoder outputs
            z_s, p_s,       # simsiam outputs (source)
            z_t, p_t,       # simsiam outputs (target)
            svdd_feat_s,    # deep_svdd feature (from z_s)
            svdd_feat_t     # deep_svdd feature (from z_t)
        )