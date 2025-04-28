# models/resnet_ae.py

import torch
import torch.nn as nn
from .modules.encoder import ResNet50Encoder
from .modules.decoder import ImageDecoder

class ResNetAE(nn.Module):
    """
    AutoEncoder using ResNet50 as encoder and ImageDecoder as decoder for image reconstruction.
    Takes two domain inputs: source (s) and target (t).
    Args:
        channels (int): Number of output channels for the decoder. Default is 3.
        height (int): Height of the output image.
        width (int): Width of the output image.
        freeze (bool): Whether to freeze ResNet50 weights. Default is True.
        dec_latent_dim (int): Dimension of the latent space. Default is 2048.
        dec_hidden_dims (list): List of channel dimensions for decoder's upsampling stages.
        dropout (float): Dropout probability for the decoder. Default is 0.0.
        use_batchnorm (bool): Whether to use batch normalization in the decoder. Default is False.
    
    Note:
        Ensure that height and width are divisible by 2 ** (len(dec_hidden_dims) - 1).
        For optimal performance, preprocess input images to 224x224 and apply ImageNet normalization
        (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]).
    """
    def __init__(self,
                 channels: int = 3,
                 height: int = None,
                 width: int = None,
                 enc_freeze: bool = True,
                 dec_latent_dim: int = 2048,
                 dec_hidden_dims: list = None,
                 dropout: float = 0.0,
                 use_batchnorm: bool = False
                 ):
        super(ResNetAE, self).__init__()
        if dec_hidden_dims is None:
            raise ValueError("dec_hidden_dims must be provided as a list")
        if height is None or width is None:
            raise ValueError("height and width must be specified")
        self.encoder = ResNet50Encoder(latent_dim=dec_latent_dim, freeze=enc_freeze)
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
        Forward pass of the AutoEncoder.
        Args:
            x_s (torch.Tensor): Source domain input image tensor of shape (batch_size, 3, height, width)
            x_t (torch.Tensor): Target domain input image tensor of shape (batch_size, 3, height, width)
        """
        e_s = self.encoder(x_s)
        e_t = self.encoder(x_t)
        x_s_recon = self.decoder(e_s)
        x_t_recon = self.decoder(e_t)

        return (
            e_s, e_t,             # encoder outputs
            x_s_recon, x_t_recon
        )