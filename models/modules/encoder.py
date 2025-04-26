# models/modules/encoder.py

import torch
import torch.nn as nn
from torchvision import models


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
        use_batchnorm (bool): whether to use batch normalization
    """
    def __init__(self, in_dim, hidden_dims, latent_dim, dropout=0.0, use_batchnorm=False):
        super(MLPEncoder, self).__init__()
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
        dropout=0.0,
        use_batch_norm=False
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

class ResNet50Encoder(nn.Module):
    """
    Encoder for feature extraction from image data using pretrained ResNet50.
    Args:
        latent_dim (int, optional): Dimension of the output feature vector. If None, uses ResNet50's default feature dimension (2048).
        freeze (bool): Whether to freeze ResNet50 weights. Defaults to True.
    
    Input shape: (batch_size, 3, height, width)
    Output shape: (batch_size, latent_dim) or (batch_size, 2048) if latent_dim is None
    
    Note:
        For optimal performance, resize input images to 224x224 and apply ImageNet normalization
        (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]).
    """
    def __init__(self, latent_dim=None, freeze=True):
        super(ResNet50Encoder, self).__init__()
        # Load pretrained ResNet50
        resnet = models.resnet50(weights='DEFAULT')
        
        # Remove the final fully connected layer for feature extraction
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        
        # Freeze weights if specified
        if freeze:
            for param in self.features.parameters():
                param.requires_grad = False
            self.features.eval()
        
        # Store ResNet50 feature dimension
        self.feature_dim = resnet.fc.in_features  # 2048
        
        # Add linear layer to adjust output dimension if specified
        if latent_dim is not None:
            self.fc = nn.Linear(self.feature_dim, latent_dim)
        else:
            self.fc = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the encoder.
        Args:
            x (torch.Tensor): Input image tensor of shape (batch_size, 3, height, width)
        Returns:
            torch.Tensor: Feature tensor of shape (batch_size, latent_dim) or (batch_size, 2048)
        """
        # Extract features using ResNet50
        x = self.features(x)  # Shape: (batch_size, 2048, 1, 1)
        x = x.view(x.size(0), -1)  # Shape: (batch_size, 2048)
        
        # Apply linear layer if specified
        if self.fc is not None:
            x = self.fc(x)  # Shape: (batch_size, latent_dim)
        return x