# models/resnet_sim_svdd_ae_v2.py

import torch
import torch.nn as nn
from .modules.encoder import ResNet50Encoder
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
                 ae_use_batchnorm: bool = False
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

    def forward(self, x_s: torch.Tensor, x_t: torch.Tensor):
        e_s = self.encoder(x_s)
        e_t = self.encoder(x_t)

        x_s_recon = self.decoder(e_s)
        x_t_recon = self.decoder(e_t)

        return e_s, e_t, x_s_recon, x_t_recon
    