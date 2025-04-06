# models/modules/encoder.py

import torch
import torch.nn as nn
import torch.nn.functional as F

from .tcn import TemporalBlock

class BaseEncoder(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        raise NotImplementedError

class MLPEncoder(BaseEncoder):
    """
    MLP Encoder.
    input shape: (batch_size, in_channels, seq_len)
    output: (batch_size, latent_dim)
    Args:
        in_dim (int): The dimension after flattening (C*T)
        hidden_dims (list): List of intermediate layer sizes
        latent_dim (int): Final latent dimension
        dropout (float): Dropout rate
    """
    def __init__(self, in_dim, hidden_dims, latent_dim, dropout=0.0):
        super(MLPEncoder, self).__init__()
        layers = []
        in_dim = in_dim

        # Hidden layer stack
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = h_dim

        # output: latent dim layer
        layers.append(nn.Linear(in_dim, latent_dim))

        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)  # (B, C*T)
        return self.mlp(x)         # (B, latent_dim)

class TCNEncoder(BaseEncoder):
    """
    TCN(Temporal Convolutional Network) Block Encoder
    input: (batch_size, in_channels, seq_len)
    output: (B, out_ch, out_seq_len)
    Args:
        in_channels (int): Size of the input channels (e.g., 2)
        n_layers (int): Number of TCN blocks (TemporalBlocks)
        filters_base (int): out_channels of the first block
        filters_factor (int): Factor by which channels increase per block (e.g., 2)
        kernel_size (int): Kernel size to be used in the TCN (default=3)
        stride (int): stride to be used in each TemporalBlock (default: conv1= 2, conv2= 1)
        dilation_base (int): Base multiplier for calculating dilation (default=2 → 1, 2, 4, ...)
        use_batch_norm (bool): Whether to use BatchNorm
        dropout (float): Dropout rate
    """
    def __init__(
        self,
        in_channels=2,
        n_layers=3,
        filters_base=4,
        filters_factor=2,
        kernel_size=3,
        stride=2,
        dilation_base=2,
        use_batch_norm=False,
        dropout=0.0
    ):
        super(TCNEncoder, self).__init__()
        
        layers = []
        prev_channels = in_channels

        # Stack multiple TemporalBlock instances
        for i in range(n_layers):
            out_channels = filters_base * (filters_factor ** i)
            dilation = dilation_base ** i

            block = TemporalBlock(
                in_channels=prev_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                dilation=dilation,
                stride=stride,
                use_batch_norm=use_batch_norm,
                dropout=dropout
            )
            layers.append(block)
            prev_channels = out_channels

        # TCN composed of a stack
        self.tcn = nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.tcn(x)  # (B, out_ch, out_seq_len)
    
