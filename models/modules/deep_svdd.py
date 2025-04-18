# models/modules/deep_svdd.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class DeepSVDD(nn.Module):
    """
    One-Class Deep SVDD Module
    Args:
        backbone (nn.Module): Feature extraction (encoding) model (e.g., SimSiam encoder + projector)
        latent_dim (int): Dimensionality of the feature (latent) space
    """
    def __init__(self, backbone: nn.Module, latent_dim: int):
        super().__init__()
        self.backbone = backbone
        self.latent_dim = latent_dim

        # Register the center vector as a buffer
        self.register_buffer("center", torch.zeros(self.latent_dim))

        # Define the radius(r) of the SVDD hypersphere as a learnable parameter
        self.radius = nn.Parameter(torch.zeros(1), requires_grad=True)
        self.radius.data.fill_(0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feature = self.backbone(x)  # shape: [batch_size, latent_dim]
        return feature

    def anomaly_score(self, x: torch.Tensor) -> torch.Tensor:
        """
        Return the anomaly score (distance) for input x
        Higher distance (dist^2) indicates a higher likelihood of being an anomaly
        """
        with torch.no_grad():
            feature = self.forward(x)
            dist = torch.sum((feature - self.center) ** 2, dim=1)  # shape: [batch_size]
        return dist

    def predict(self, x: torch.Tensor, threshold: float) -> torch.Tensor:
        """
        Determine whether input is an anomaly based on distance
        Args:
            x (torch.Tensor): Input
            threshold (float): Distance-based threshold
        Returns:
            torch.Tensor: 0 for normal, 1 for anomaly
        """
        score = self.anomaly_score(x)
        return (score > threshold).long()