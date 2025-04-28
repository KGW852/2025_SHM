# models/resnet_sim_ae.py

import torch
import torch.nn as nn
from .modules.encoder import ResNet50Encoder
from .modules.simsiam import SimSiam
from .modules.decoder import ImageDecoder

class ResNetSimAE(nn.Module):
    def __init__(self,
                 channels: int = 3,
                 height: int = None,
                 width: int = None,
                 enc_freeze: bool = False,
                 dec_latent_dim: int = 2048,
                 dec_hidden_dims: list = None,
                 dropout: float = 0.0,
                 use_batchnorm: bool = False,
                 proj_hidden_dim: int = 512,
                 proj_out_dim: int = 512,
                 pred_hidden_dim: int = 256,
                 pred_out_dim: int = 512
                 ):
        super(ResNetSimAE, self).__init__()
        if dec_hidden_dims is None:
            raise ValueError("dec_hidden_dims must be provided as a list")
        if height is None or width is None:
            raise ValueError("height and width must be specified")
        self.encoder = ResNet50Encoder(latent_dim=dec_latent_dim, freeze=enc_freeze)
        self.simsiam = SimSiam(in_dim=dec_latent_dim,
                               proj_hidden_dim=proj_hidden_dim,
                               proj_out_dim=proj_out_dim,
                               pred_hidden_dim=pred_hidden_dim,
                               pred_out_dim=pred_out_dim)
        self.decoder = ImageDecoder(
            latent_dim=dec_latent_dim,
            hidden_dims=dec_hidden_dims,
            out_channels=channels,
            height=height,
            width=width,
            dropout=dropout,
            use_batchnorm=use_batchnorm
        )

    def forward(self, x_s: torch.Tensor, x_t: torch.Tensor):
        """
        Forward pass of the ResNetSimAE.
        Args:
            x_s (torch.Tensor): Source domain input image tensor of shape (batch_size, 3, height, width)
            x_t (torch.Tensor): Target domain input image tensor of shape (batch_size, 3, height, width)
        Returns:
            e_s, e_t: Encoder outputs (batch_size, latent_dim)
            z_s, z_t: SimSiam projections (batch_size, proj_out_dim)
            p_s, p_t: SimSiam predictions (batch_size, pred_out_dim)
            x_s_recon, x_t_recon: Reconstructed images (batch_size, channels, height, width)
        """
        e_s = self.encoder(x_s)
        e_t = self.encoder(x_t)
        z_s, p_s, z_t, p_t = self.simsiam(e_s, e_t)
        x_s_recon = self.decoder(e_s)
        x_t_recon = self.decoder(e_t)
        return e_s, e_t, z_s, p_s, z_t, p_t, x_s_recon, x_t_recon