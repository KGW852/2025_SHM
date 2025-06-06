# models/criterions/svdd_loss.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class DeepSVDDLoss(nn.Module):
    """
    Deep SVDD Loss. "Deep One-Class Classification" (Ruff et al., ICML 2018)
    Args:
        nu (float): Parameter in [0, 1] controlling tolerance for outliers
        reduction (str): 'mean' or 'sum'. Determines whether the final loss for the mini-batch is averaged or summed
    """
    def __init__(self, nu: float = 0.1, reduction: str = 'mean'):
        super().__init__()
        assert reduction in ['mean', 'sum', 'simple'], f"Invalid reduction mode: {reduction}"
        self.nu = nu
        self.reduction = reduction

    def forward(self, features: torch.Tensor, center: torch.Tensor, radius: torch.nn.Parameter) -> torch.Tensor:
        """
        Args:
            features (torch.Tensor): Model's forward output (features in latent space), shape: [batch_size, latent_dim]
            center (torch.Tensor): Center parameter of the SVDD hypersphere (fixed initially or updated during training), shape: [latent_dim]
            radius (nn.Parameter): Radius of the SVDD hypersphere, updated during training, shape: []
        Returns:
            torch.Tensor: deep svdd loss (scalar)
        """
        dist_sq = torch.sum((features - center) ** 2, dim=1)  # [batch_size]
        
        if self.reduction == 'simple':
            loss = torch.mean(dist_sq)  # loss = torch.mean(dist_sq) / features.size(1)
        else:  # Ruff et al. (ICML 2018): Cumulative form, L = R^2 + (1 / (nu * N)) * sum( max(0, dist_sq - R^2) )
            dist_diff = dist_sq - radius.pow(2)
            loss_term = torch.clamp(dist_diff, min=0)
            if self.reduction == 'mean':
                loss = radius.pow(2) + (1.0 / (self.nu * features.size(0))) * torch.sum(loss_term)
            elif self.reduction == 'sum':
                loss = radius.pow(2) + (1.0 / self.nu) * torch.sum(loss_term)
            else:
                raise ValueError(f"Not supported reduction mode: {self.reduction}")
        return loss