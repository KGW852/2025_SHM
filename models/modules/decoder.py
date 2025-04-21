# models/modules/decoder.py

import torch
import torch.nn as nn
import torch.nn.functional as F

from .tcn import TemporalBlock

class BaseDecoder(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        raise NotImplementedError

class MLPDecoder(BaseDecoder):
    """
    MLP Decoder.
    input shape: (batch_size, latent_dim)
    output: (batch_size, out_channels, out_seq_len)
    Args:
        latent_dim (int): Latent space dimension (output size from the Encoder)
        hidden_dims (list): List of intermediate layer sizes
        out_channels (int): Number of final channels to reconstruct
        out_seq_len (int): Final time axis length to reconstruct
        dropout (float): Dropout rate
        use_batchnorm (bool): whether to use batch normalization
    """
    def __init__(self, latent_dim, hidden_dims, out_channels, out_seq_len, dropout=0.0, use_batchnorm=False):
        super(MLPDecoder, self).__init__()
        self.out_channels = out_channels
        self.out_seq_len = out_seq_len
        
        layers = []
        prev_dim = latent_dim

        # hidden layers
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            if use_batchnorm:
                layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = h_dim

        # output: out layer
        out_dim = out_channels * out_seq_len
        layers.append(nn.Linear(prev_dim, out_dim))

        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.mlp(x)                                             # (B, out_channels*out_seq_len)
        x = x.view(x.size(0), self.out_channels, self.out_seq_len)  # reshape: (B, out_channels, out_seq_len)
        return x
