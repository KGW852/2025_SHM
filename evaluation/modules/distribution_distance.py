# evaluation/modules/distribution_distance.py

import torch

def get_boundary(embeddings: torch.Tensor, center: torch.Tensor, percentile: int = 99) -> float:
    """
    Calculate the L2 distance from the given 'center' for each embedding,
    then return the specified percentile value as threshold.
    Args
        embeddings (torch.Tensor): (N, feature_dim) Embedding vectors for N samples
        percentile (int): 99 means the value at the 99th percentile of the distance distribution is used as the boundary
    Returns
        float: The set threshold (values above this suggest a high likelihood of being an anomaly)
    """
    # compute the L2 distance of each embedding from the mean
    distances = torch.norm(embeddings - center, dim=1)  # (N,)

    # set the 'percentile' value from the distance distribution as the threshold
    threshold = distances.quantile(q=percentile / 100.0).item()

    return threshold
