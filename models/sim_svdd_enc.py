# models/sim_svdd_enc.py

import torch
import torch.nn as nn
from .modules.encoder import MLPEncoder
from .modules.simsiam import SimSiam
from .modules.deep_svdd import DeepSVDD, SVDDBackbone

class SimSVDDEnc(nn.Module):
    """
    Combine MLPEncoder and SimSiam for source/target domain inputs.
    then pass z_s, z_t to DeepSVDD for additional feature processing (or identity).
    Forward:
        x_s, x_t: (B, C, T)
    Returns:
        e_s, e_t: encoder outputs (latent_dim).
        z_s, p_s, z_t, p_t: SimSiam projector/predictor results.
        feat_s, feat_t: DeepSVDD feature outputs (from z_s, z_t)
        dist_s, dist_t: L2 squared distance to the SVDD center (for One-Class Objective)
    """
    def __init__(self,
                 enc_in_dim: int,
                 enc_hidden_dims: list,
                 enc_latent_dim: int,
                 enc_dropout: float = 0.0,
                 enc_use_batchnorm: bool = False,
                 proj_hidden_dim: int = 64,
                 proj_out_dim: int = 64,
                 pred_hidden_dim: int = 32,
                 pred_out_dim: int = 64,
                 svdd_in_dim: int = 64,
                 svdd_hidden_dims: list = [32],
                 svdd_latent_dim: int = 32,
                 svdd_dropout: float = 0.0,
                 svdd_use_batchnorm: bool = False
                 ):
        super(SimSVDDEnc, self).__init__()

        # MLP Encoder
        self.encoder = MLPEncoder(
            in_dim=enc_in_dim,
            hidden_dims=enc_hidden_dims,
            latent_dim=enc_latent_dim,
            dropout=enc_dropout,
            use_batchnorm=enc_use_batchnorm)
        
        # SimSiam
        self.simsiam = SimSiam(
            in_dim=enc_latent_dim,
            proj_hidden_dim=proj_hidden_dim,
            proj_out_dim=proj_out_dim,
            pred_hidden_dim=pred_hidden_dim,
            pred_out_dim=pred_out_dim)
        
        # DeepSVDD backbone
        self.svdd_backbone = SVDDBackbone(
            in_dim=svdd_in_dim,
            hidden_dims=svdd_hidden_dims,
            latent_dim=svdd_latent_dim,
            dropout=svdd_dropout,
            use_batchnorm=svdd_use_batchnorm)

        # DeepSVDD
        self.svdd = DeepSVDD(
            backbone=self.svdd_backbone,
            latent_dim=svdd_latent_dim)

    def forward(self, x_s: torch.Tensor, x_t: torch.Tensor):
        e_s = self.encoder(x_s)  # (B, latent_dim)
        e_t = self.encoder(x_t)

        z_s, p_s, z_t, p_t = self.simsiam(e_s, e_t)

        feat_s, dist_s = self.svdd(z_s)  # (B, svdd_latent_dim), (B,)
        feat_t, dist_t = self.svdd(z_t)

        return (
            e_s, e_t,        # encoder outputs
            z_s, p_s,        # simsiam outputs (source)
            z_t, p_t,        # simsiam outputs (target)
            feat_s, feat_t,  # svdd feature (from z_s, z_t)
            dist_s, dist_t   # distance to svdd center
        )