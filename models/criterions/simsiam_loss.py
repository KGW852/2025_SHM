# models/criterions/simsiam_loss.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class SimSiamLoss(nn.Module):
    """
    SimSiam Loss
    reference: https://arxiv.org/abs/2011.10566
    """
    def __init__(self):
        super(SimSiamLoss, self).__init__()
        
    @staticmethod
    def _neg_cosine_similarity(p, z):
        """
        Negative cosine similarity
        p, z: [batch_size, feat_dim]
        """
        # normalize p, z
        p = F.normalize(p, dim=1)
        z = F.normalize(z, dim=1)
        return -(p * z).sum(dim=1).mean()

    def forward(self, p1, z2, p2, z1):
        """
        z1, p1: first projection, prediction
        z2, p2: second projection, prediction
        
        z2, z1: stop gradient to use .detach()
        """
        loss_12 = self._neg_cosine_similarity(p1, z2.detach())
        loss_21 = self._neg_cosine_similarity(p2, z1.detach())
        
        loss = 0.5 * (loss_12 + loss_21)
        return loss