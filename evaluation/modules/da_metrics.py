# evaluation/modules/da_metrics.py

import torch
from typing import Optional

from utils.csv_utils import save_csv


def mmd_score(e_s: torch.Tensor, e_t: torch.Tensor, kernel: str = 'rbf', sigma: float = None) -> torch.Tensor:
    """
    Compute the Maximum Mean Discrepancy (MMD) between the latent features of domains.
    Args:
        e_s (torch.Tensor): source domain latent feature (shape: [N, feature_dim])
        e_t (torch.Tensor): target domain latent feature (shape: [M, feature_dim])
        kernel (str): Kernel type (support 'rbf', default 'rbf')
        sigma (float, optional): Bandwidth (standard deviation) of the RBF kernel. If None, it is automatically estimated based on the data.
    Returns:
        torch.Tensor: Computed MMD values (scalar tensor)
    """
    assert kernel == 'rbf'
    N, M = e_s.size(0), e_t.size(0)

    # 1. sigma automatic estimation (median heuristic)
    if sigma is None:
        all_features = torch.cat([e_s, e_t], dim=0)
        pairwise_dists = torch.pdist(all_features)  # 1-D tensor of distances for all combinations
        sigma = pairwise_dists.median().item()
        if sigma <= 0:
            sigma = 1e-6
    sigma = float(sigma)
    
    # 2. compute all kernel values ​​(source+target combined)
    # distance matrix: (N+M)x(N+M)
    all_features = torch.cat([e_s, e_t], dim=0)
    dist_matrix = torch.cdist(all_features, all_features, p=2)  # L2 distance matrix
    K = torch.exp(-(dist_matrix ** 2) / (2 * (sigma ** 2)))     # RBF kernel matrix
    # Split the kernel matrix into parts
    K_xx = K[:N, :N]    # source-source kernel value (NxN)
    K_yy = K[N:, N:]    # target-target kernel value (MxM)
    K_xy = K[:N, N:]    # source-target kernel value (NxM)

    # 3. compute MMD
    if N > 1:
        # Sum excluding diagonal elements (since all diagonals are 1, remove them with sum - trace)
        sum_xx = K_xx.sum() - torch.sum(torch.diag(K_xx))
        mmd_xx = sum_xx / (N * (N - 1))
    else:
        mmd_xx = 0.0

    if M > 1:
        sum_yy = K_yy.sum() - torch.sum(torch.diag(K_yy))
        mmd_yy = sum_yy / (M * (M - 1))
    else:
        mmd_yy = 0.0

    mmd_xy = K_xy.sum() / (N * M) if (N > 0 and M > 0) else 0.0
    mmd_value = mmd_xx + mmd_yy - 2 * mmd_xy  # MMD score
    return mmd_value if isinstance(mmd_value, torch.Tensor) else torch.tensor(mmd_value)

class DAMetric:
    def __init__(self):
        self.mmd_value: Optional[float] = None

    def calc_metric(self, e_s: torch.Tensor, e_t: torch.Tensor, sigma: Optional[float] = None) -> float:
        mmd_tensor = mmd_score(e_s, e_t, kernel="rbf", sigma=sigma)

        if mmd_tensor.numel() != 1:
            raise ValueError("mmd_score result is not a scalar.")
        self.mmd_value = mmd_tensor.item()

        return self.mmd_value

    def save_metric_as_csv(self, metric_csv_path):
        if self.mmd_value is None:
            raise ValueError("MMD value does not exist. Call calc_metric first.")
        
        # save metric to csv
        rows = [["metric", "value"], ["MMD", self.mmd_value]]

        save_csv(rows, metric_csv_path)
        