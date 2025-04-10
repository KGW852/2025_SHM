# evaluation/modules/cos_nn_distance.py

import torch
import torch.nn.functional as F
from typing import Dict

def compute_cross_nn_distance(source_embeddings: torch.Tensor, target_embeddings: torch.Tensor) -> Dict[str, float]:
    """
    Compute cross-domain nearest neighbor distance between source and target embeddings using cosine distance = (1 - cosine_similarity).
    Args:
        source_embeddings (torch.Tensor): shape (num_source, feature_dim)
        target_embeddings (torch.Tensor): shape (num_target, feature_dim)
        distance_metric (str): distance metric to use, e.g., 'euclidean', 'cosine', supported by scipy.spatial.distance.cdist
    Returns:
        result_dict (dict):
            'mean_s2t': float,        # Mean nearest neighbor distance from source to target
            'median_s2t': float,      # Median nearest neighbor distance from source to target
            'mean_t2s': float,        # Mean nearest neighbor distance from target to source
            'median_t2s': float,      # Median nearest neighbor distance from target to source
            'mean_cross_nn': float,   # (mean_s2t + mean_t2s) / 2
            'median_cross_nn': float  # (median_s2t + median_t2s) / 2
    """
    # Pairwise cosine similarity from source to target (N x M)
    # (N: number of source samples, M: number of target samples)
    # shape: (num_source, num_target)
    cos_sim_s2t = F.cosine_similarity(
        source_embeddings.unsqueeze(1),  # (N, 1, D)
        target_embeddings.unsqueeze(0),  # (1, M, D)
        dim=-1
    )
    
    # Cosine distance = 1 - cosine similarity
    dist_s2t = 1.0 - cos_sim_s2t  # (N, M)

    # Nearest neighbor distance in the target domain for each source sample (min over M)
    nearest_s2t = dist_s2t.min(dim=1).values  # shape: (N,)
    mean_s2t = nearest_s2t.mean().item()
    median_s2t = nearest_s2t.median().item()
    
    # Target-to-source can also be computed by transposing dist_s2t and applying min
    # dist_s2t.transpose(0, 1) => (M, N)
    nearest_t2s = dist_s2t.min(dim=0).values  # shape: (M,)
    mean_t2s = nearest_t2s.mean().item()
    median_t2s = nearest_t2s.median().item()
    
    # cross NN distance (mean/median)
    mean_cross_nn = 0.5 * (mean_s2t + mean_t2s)
    median_cross_nn = 0.5 * (median_s2t + median_t2s)
    
    result_dict = {
        "mean_s2t": mean_s2t,
        "median_s2t": median_s2t,
        "mean_t2s": mean_t2s,
        "median_t2s": median_t2s,
        "mean_cross_nn": mean_cross_nn,
        "median_cross_nn": median_cross_nn
    }
    return result_dict

"""
if __name__ == "__main__":
    torch.manual_seed(42)
    
    # example: 5dim vectors, source 10EA, target 8EA.
    source_emb = torch.randn(10, 5)
    target_emb = torch.randn(8, 5)
    
    result = compute_cross_nn_distance(source_emb, target_emb)
    for k, v in result.items():
        print(f"{k}: {v}")
"""
