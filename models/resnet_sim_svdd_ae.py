# models/resnet_sim_svdd_ae.py

import torch
import torch.nn as nn
from .modules.encoder import ResNet50Encoder
from .modules.simsiam import SimSiam
from .modules.deep_svdd import SVDDBackbone, DeepSVDD
from .modules.decoder import ImageDecoder


class ResNetSimSVDDAE(nn.Module):
    def __init__(self,
                 # AE
                 channels: int = 3,
                 height: int = None,
                 width: int = None,
                 enc_freeze: bool = False,
                 dec_latent_dim: int = 2048,
                 dec_hidden_dims: list = None,
                 ae_dropout: float = 0.0,
                 ae_use_batchnorm: bool = False,
                 # SimSiam
                 proj_hidden_dim: int = 512,
                 proj_out_dim: int = 512,
                 pred_hidden_dim: int = 256,
                 pred_out_dim: int = 512,
                 # SVDD
                 svdd_in_dim: int = 512,
                 svdd_hidden_dims: list = [256],
                 svdd_latent_dim: int = 256,
                 svdd_center_param: bool = False,
                 svdd_radius_param: bool = False,
                 svdd_dropout: float = 0.0,
                 svdd_use_batchnorm: bool = False
                 ):
        super(ResNetSimSVDDAE, self).__init__()

        # ResNet Encoder
        self.encoder = ResNet50Encoder(latent_dim=dec_latent_dim, freeze=enc_freeze)

        # ResNet Decoder
        self.decoder = ImageDecoder(
            latent_dim=dec_latent_dim,
            hidden_dims=dec_hidden_dims,
            out_channels=channels,
            height=height,
            width=width,
            dropout=ae_dropout,
            use_batchnorm=ae_use_batchnorm)

        # SimSiam
        self.simsiam = SimSiam(in_dim=dec_latent_dim,
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
            backbone=nn.Identity(),
            enc_latent_dim=dec_latent_dim,
            latent_dim=svdd_latent_dim,
            center_param=svdd_center_param,
            radius_param=svdd_radius_param)

    def forward(self, x_s: torch.Tensor, x_t: torch.Tensor):
        e_s = self.encoder(x_s)
        e_t = self.encoder(x_t)

        z_s, p_s, z_t, p_t = self.simsiam(e_s, e_t)

        feat_s = self.svdd(z_s)  # (B, svdd_latent_dim), (B,)
        feat_t = self.svdd(z_t)

        x_s_recon = self.decoder(e_s)
        x_t_recon = self.decoder(e_t)

        return e_s, e_t, z_s, p_s, z_t, p_t, feat_s, feat_t, x_s_recon, x_t_recon
    