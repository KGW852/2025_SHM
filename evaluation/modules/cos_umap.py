# evaluation/modules/cos_umap.py

import torch
import numpy as np
from typing import Dict, Any

import umap.umap_ as umap

from .cos_nn_distance import compute_cross_nn_distance


class CosUMAP:
    """
    PyTorch-based UMAP (PUMAP) class that embeds high-dimensional features (from two domains)
    using cosine distance (metric='cosine') and measures similarity/distance between distributions
    in the embedded space.
    Args:
        n_neighbors (int): Number of nearest neighbors for UMAP
        n_components (int): Number of dimensions to embed into with UMAP
        model (PUMAP): PUMAP model object
    """
    def __init__(self, n_neighbors, n_components, random_state):
        self.n_neighbors = n_neighbors
        self.n_components = n_components
        self.random_state = random_state

        self.model = PUMAP(
            n_neighbors=n_neighbors,
            n_components=n_components,
            metric='cosine',
            random_state=random_state
        )

    def fit_transform(self, source_features: torch.Tensor, target_features: torch.Tensor):
        """
        Fit the PUMAP model by combining features from two domains (source, target)
        return the embedding results mapped to a lower-dimensional space.
        Args:
            source_features (torch.Tensor): Source domain features [N, D]
            target_features (torch.Tensor): Target domain features [M, D]
        """
        combined = torch.cat([source_features, target_features], dim=0)
        embedding = self.model(combined)

        source_size = source_features.size(0)
        source_embedded = embedding[:source_size]  # [N, n_components]
        target_embedded = embedding[source_size:]

        return source_embedded, target_embedded