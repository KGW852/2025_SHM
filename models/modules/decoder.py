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

class UpsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_batchnorm=False, dropout=0.0):
        super(UpsampleBlock, self).__init__()
        self.trans_conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)
        self.bn = nn.BatchNorm2d(out_channels) if use_batchnorm else None
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

    def forward(self, x):
        x = self.trans_conv(x)
        if self.bn is not None:
            x = self.bn(x)
        x = self.relu(x)
        if self.dropout is not None:
            x = self.dropout(x)
        return x

class ImageDecoder(nn.Module):
    """
    Decoder for reconstructing images from latent space using transposed convolutions.
    Args:
        latent_dim (int): Dimension of the input latent vector.
        hidden_dims (list): List of channel dimensions for each upsampling stage.
        out_channels (int): Number of output channels (e.g., 3 for RGB).
        height (int): Height of the output image.
        width (int): Width of the output image.
        dropout (float): Dropout probability. Defaults to 0.0.
        use_batchnorm (bool): Whether to use batch normalization. Defaults to False.
    
    Input shape: (batch_size, latent_dim)
    Output shape: (batch_size, out_channels, height, width)
    
    Note:
        The height and width must be divisible by 2 ** (len(hidden_dims) - 1).
    """
    def __init__(self, latent_dim, hidden_dims, out_channels, height, width, dropout=0.0, use_batchnorm=False):
        super(ImageDecoder, self).__init__()
        if len(hidden_dims) < 1:
            raise ValueError("hidden_dims must have at least one element")
        m = len(hidden_dims) - 1
        if height % (2 ** m) != 0 or width % (2 ** m) != 0:
            raise ValueError(f"height and width must be divisible by 2 ** (len(hidden_dims) - 1) = {2 ** m}")
        h0 = height // (2 ** m)
        w0 = width // (2 ** m)
        self.fc = nn.Linear(latent_dim, hidden_dims[0] * h0 * w0)
        self.upsample_blocks = nn.ModuleList([
            UpsampleBlock(hidden_dims[i], hidden_dims[i+1], use_batchnorm, dropout)
            for i in range(m)
        ])
        self.final_layer = nn.Conv2d(hidden_dims[-1], out_channels, kernel_size=3, stride=1, padding=1)
        self.h0 = h0
        self.w0 = w0
        self.hidden_dims = hidden_dims

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the decoder.
        Args:
            z (torch.Tensor): Latent vector of shape (batch_size, latent_dim)
        Returns:
            torch.Tensor: Reconstructed image of shape (batch_size, out_channels, height, width)
        """
        x = self.fc(z)
        x = x.view(-1, self.hidden_dims[0], self.h0, self.w0)
        for block in self.upsample_blocks:
            x = block(x)
        x = self.final_layer(x)
        x = torch.sigmoid(x)
        return x