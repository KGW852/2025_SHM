# models/modules/deep_svdd.py

import torch
import torch.nn as nn


class SVDDBackbone(nn.Module):
    """
    MLP backbone for DeepSVDD.
    Args:
        in_dim (int): input dimension (e.g., from encoder output: z.shape[-1])
        hidden_dims (list of int): list of hidden layer sizes
        latent_dim (int): final embedding dimension for SVDD
        dropout (float): dropout probability
        use_batchnorm (bool): whether to use batch normalization
    """
    def __init__(self, in_dim: int, hidden_dims: list, latent_dim: int, dropout=0.0, use_batchnorm=False):
        super().__init__()
        layers = []
        prev_dim = in_dim

        # hidden layers
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            if use_batchnorm:
                layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = h_dim
        
        # output: latent layer
        layers.append(nn.Linear(prev_dim, latent_dim))

        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)

class DeepSVDD(nn.Module):
    """
    One-Class Deep SVDD Module
    Args:
        backbone (nn.Module): Feature extraction (encoding) model (e.g., SimSiam encoder + projector)
        enc_latent_dim (int): Dimensionality of the embedded (latent) space (e.g., encoder output)
        latent_dim (int): Dimensionality of the feature (latent) space
        center_param (bool): If True, center is a learnable parameter; else, it's a buffer
    """
    def __init__(self, backbone: nn.Module, enc_latent_dim: int, latent_dim: int, center_param: bool = False, radius_param: bool = False):
        super(DeepSVDD, self).__init__()
        self.backbone = backbone
        self.enc_latent_dim = enc_latent_dim
        self.latent_dim = latent_dim
        self.center_param = center_param
        self.radius_param = radius_param

        if self.center_param:  # define center as a learnable parameter
            self.center_enc = nn.Parameter(torch.zeros(self.enc_latent_dim), requires_grad=True)
            self.center_feat = nn.Parameter(torch.zeros(self.latent_dim), requires_grad=True)            
        else:  # register center as a buffer
            self.register_buffer('center_enc', torch.zeros(self.enc_latent_dim))
            self.register_buffer('center_feat', torch.zeros(self.latent_dim))
            
        if self.radius_param:  # Define radius as a learnable parameter
            self.radius = nn.Parameter(torch.tensor(1.0), requires_grad=True)
        else:  # Register radius as a buffer
            self.register_buffer('radius', torch.tensor(1.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.backbone(x)  # (B, latent_dim)
        return feat
