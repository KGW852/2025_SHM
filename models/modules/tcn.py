# models/modules/tcn.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm

class TemporalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=2,
                 use_batch_norm=False, dropout=0.0):
        """
        Temporal Convolutional Network encoder block.
        Args:
          in_channels:    Number of input channels.
          out_channels:   Number of output channels.
          kernel_size:    Convolution kernel size (default=3).
          dilation:       Dilation for the first conv (second conv will use dilation*2 by default).
          stride:         Stride for temporal downsampling (default=1; use 2 or more to reduce sequence length).
          use_batch_norm: If True, applies BatchNorm1d after convolutions (default=False).
          dropout:        Dropout probability (default=0.0).
        """
        super(TemporalBlock, self).__init__()
        # Compute padding for causal convolutions. 
        # For a causal conv, we pad (kernel_size-1)*dilation on the left so that outputs do not depend on future inputs.
        self.pad1 = (kernel_size - 1) * dilation               # left padding for first conv
        self.pad2 = (kernel_size - 1) * (dilation * 2)         # left padding for second conv (dilation doubled)
        
        # First dilated causal Conv1d layer with weight normalization (and optional BatchNorm).
        # Use stride > 1 to downsample the time dimension.
        self.conv1 = weight_norm(nn.Conv1d(
            in_channels, out_channels, kernel_size,
            stride=stride, dilation=dilation, bias=not use_batch_norm))
        self.bn1 = nn.BatchNorm1d(out_channels) if use_batch_norm else None
        
        # Second dilated causal Conv1d layer (dilation doubled) with weight normalization.
        # No downsampling on this layer (stride=1).
        self.conv2 = weight_norm(nn.Conv1d(
            out_channels, out_channels, kernel_size,
            stride=1, dilation=dilation * 2, bias=not use_batch_norm))
        self.bn2 = nn.BatchNorm1d(out_channels) if use_batch_norm else None
        
        # Activation and dropout (applied after each convolution).
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else None
        
        # Residual connection: 1x1 convolution for matching channel count and downsampling time if needed.
        if stride != 1 or in_channels != out_channels:
            # If the input and output dimensions differ (in channels or length), 
            # use 1x1 conv with the same stride to adjust the residual.
            self.downsample = nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride)
        else:
            self.downsample = None

    def forward(self, x):
        """
        Forward pass for the TCNEncoderBlock.
        Input: x of shape (batch, in_channels, seq_len)
        Output: tensor of shape (batch, out_channels, seq_len_out) where seq_len_out = ceil(seq_len/stride).
        """
        # Save identity (skip connection)
        identity = x

        # First convolution with left padding for causality
        out = F.pad(x, (self.pad1, 0))             # pad left side
        out = self.conv1(out)                      # dilated causal conv
        if self.bn1 is not None:
            out = self.bn1(out)                    # optional batch normalization
        out = self.activation(out)                 # ReLU activation
        if self.dropout is not None:
            out = self.dropout(out)                # optional dropout
        
        # Second convolution (on the output of first conv) with its own left padding
        out = F.pad(out, (self.pad2, 0))           # pad left side for second conv
        out = self.conv2(out)                      # second dilated conv
        if self.bn2 is not None:
            out = self.bn2(out)                    # optional batch normalization
        out = self.activation(out)                 # ReLU activation
        if self.dropout is not None:
            out = self.dropout(out)                # optional dropout
        
        # Residual/skip connection: downsample original input if needed to match out's shape
        if self.downsample is not None:
            identity = self.downsample(identity)   # adjust channels and length of identity
        
        # Add the residual (skip connection) to the output of conv layers
        out += identity
        
        return out
