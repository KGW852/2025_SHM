# models/modules/recon_loss.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class ReconLoss(nn.Module):
    def __init__(self, loss_type: str = 'mse', reduction: str = 'mean'):
        super(ReconLoss, self).__init__()
        assert loss_type in ['mse', 'mae'], "loss_type should be either 'mse' or 'mae'"
        
        self.loss_type = loss_type
        self.reduction = reduction

    def forward(self, x_recon: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        if self.loss_type == 'mse':
            loss = F.mse_loss(x_recon, x, reduction=self.reduction)
        else:  # self.loss_type == 'mae'
            loss = F.l1_loss(x_recon, x, reduction=self.reduction)
        
        return loss