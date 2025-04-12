# evaluation/modules/umap.py

import numpy as np
import torch
import umap
import matplotlib.pyplot as plt

from typing import Dict, List, Tuple, Union, Optional

class UMAP:
    """
    A tool for visualizing high-dimensional embeddings from multiple domains using UMAP with distance.
    
    This class combines embeddings from multiple domains (in the same latent space) and projects them to 2D 
    using UMAP (Uniform Manifold Approximation and Projection) with distance as the similarity metric.
    """
    def __init__(self, 
                 n_neighbors: int = 15, 
                 min_dist: float = 0.1, 
                 n_components: int = 2,
                 random_state: Optional[int] = None,
                 metric: str = None,
                 **umap_kwargs):
        """
        Initialize the CosineUMAP visualizer with UMAP parameters.
        Args:
            n_neighbors: Number of neighbors to consider in UMAP (controls local/global balance).
            min_dist: Minimum distance apart in the low-dimensional space (controls clustering tightness).
            n_components: Dimensionality of reduced embedding (2 for 2D plot, can be 1 or 3 as well).
            random_state: Seed for random number generator (for UMAP initialization and reproducibility).
            metric: Distance metric to use in UMAP. (default: 'None')
            **umap_kwargs: Additional keyword arguments to pass to the umap.UMAP constructor.
        """
        self.n_neighbors = n_neighbors
        self.min_dist = min_dist
        self.n_components = n_components
        self.random_state = random_state
        self.metric = metric  # Always use metric for distance in high-dimensional space
        self.umap_kwargs = umap_kwargs  # Store any additional UMAP parameters provided
    
    def fit_transform(self, embeddings: Union[Dict[str, torch.Tensor], List[torch.Tensor]], labels: Optional[List[str]] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Combine embeddings from multiple domains and perform UMAP dimensionality reduction to 2D.
        Args:
            embeddings: A dictionary mapping each domain label to a PyTorch tensor of shape (N, D),
                        **or** a list of PyTorch tensors [X1, X2, ...] each of shape (N_i, D).
            labels: If `embeddings` is provided as a list, `labels` must be a list of domain labels (strings or ints)
                    of the same length. If `embeddings` is a dict, this is ignored (domain labels come from the dict keys).
        Returns:
            A tuple (embeddings_2d, label_array):
            embeddings_2d: NumPy array of shape (total_points, 2) with UMAP 2D coordinates for all input points.
            label_array: NumPy array or list of length total_points, containing the domain label for each point.
        """
        # Determine input format and prepare domain-labeled data
        if isinstance(embeddings, dict):
            # Dictionary of {label: tensor} format
            data_items = list(embeddings.items())
        elif isinstance(embeddings, list):
            if labels is None or len(labels) != len(embeddings):
                raise ValueError("When providing embeddings as a list, a list of labels of the same length must be provided.")
            data_items = list(zip(labels, embeddings))
        else:
            raise ValueError("Embeddings must be a dict of {label: tensor} or a list of tensors with a labels list.")
        
        # Verify and collect data
        combined_data_list: List[np.ndarray] = []
        combined_labels: List[str] = []
        expected_dim: Optional[int] = None
        
        for domain_label, tensor in data_items:
            # Ensure tensor is a PyTorch tensor and move to CPU if it's on GPU, then convert to numpy
            if isinstance(tensor, torch.Tensor):
                tensor_np = tensor.detach().cpu().numpy()
            elif isinstance(tensor, np.ndarray):
                tensor_np = tensor  # already a numpy array
            else:
                # Try to convert other sequence types to numpy
                try:
                    tensor_np = np.array(tensor)
                except Exception as e:
                    raise ValueError(f"Embedding data for label '{domain_label}' is not a tensor/ndarray and cannot be converted: {e}")
            
            # Validate shape
            if tensor_np.ndim != 2:
                raise ValueError(f"Embedding data for label '{domain_label}' must be 2-dimensional, got shape {tensor_np.shape}.")
            if expected_dim is None:
                expected_dim = tensor_np.shape[1]
            elif tensor_np.shape[1] != expected_dim:
                raise ValueError(f"Dimension mismatch: domain '{domain_label}' has latent_dim {tensor_np.shape[1]}, expected {expected_dim}.")
            
            # Append data and labels
            combined_data_list.append(tensor_np)
            # Repeat the domain label for each sample in this tensor
            num_samples = tensor_np.shape[0]
            combined_labels.extend([domain_label] * num_samples)
        
        # Combine all domain data into one array
        if len(combined_data_list) == 0:
            raise ValueError("No embeddings provided for UMAP visualization.")
        combined_data = np.vstack(combined_data_list)
        
        # Configure UMAP with metric and specified parameters
        reducer = umap.UMAP(n_neighbors=self.n_neighbors,
                            min_dist=self.min_dist,
                            n_components=self.n_components,
                            metric=self.metric,
                            random_state=self.random_state,
                            **self.umap_kwargs)
        
        # Fit and transform the combined data to 2D
        embeddings_2d = reducer.fit_transform(combined_data)
        
        # Convert combined_labels to a numpy array for consistency (optional)
        label_array = np.array(combined_labels)
        
        # Store results in the object (optional, for potential later use)
        self.embedding_ = embeddings_2d
        self.labels_ = label_array
        return embeddings_2d, label_array

    def plot_embeddings(self, embeddings_umap: Optional[np.ndarray] = None, labels: Optional[np.ndarray] = None,
                        save_path: Optional[str] = None, show: bool = True) -> None:
        """
        Plot the UMAP-transformed embeddings with color-coded domain labels. Supports 1D, 2D, or 3D embeddings.
        Args:
            embeddings_umap: The UMAP-processed embedding array (shape: (n_samples, n_components)). If None, uses `self.embedding_`.
            labels: The label array of shape (n_samples,). If None, uses `self.labels_`.
            save_path: If provided, saves the figure to this path.
            show: Whether to call `plt.show()` to display the figure.
        """
        if embeddings_umap is None:
            embeddings_umap = self.embedding_
        if labels is None:
            labels = self.labels_

        if embeddings_umap is None or labels is None:
            raise ValueError("No embeddings or labels to plot. Please provide them or call fit_transform first.")

        if embeddings_umap.shape[1] not in [1, 2, 3]:
            raise ValueError(f"Can only plot 1D, 2D, or 3D embeddings, got {embeddings_umap.shape[1]}D.")

        fig = plt.figure(figsize=(8, 6))
        unique_labels = np.unique(labels)
        cmap = plt.get_cmap('tab10', len(unique_labels))

        title_str = f"UMAP projection of multi-domain embeddings ({self.metric})"

        # 1D plot
        if embeddings_umap.shape[1] == 1:
            ax = fig.add_subplot(111)
            for i, domain in enumerate(unique_labels):
                idx = (labels == domain)
                ax.scatter(
                    embeddings_umap[idx, 0],
                    np.zeros_like(embeddings_umap[idx, 0]),
                    label=domain,
                    color=cmap(i)
                )
            ax.set_xlabel("UMAP Dimension 1")
            ax.set_ylabel("Constant Zero")
            ax.set_title(title_str)

        # 2D plot
        elif embeddings_umap.shape[1] == 2:
            ax = fig.add_subplot(111)
            for i, domain in enumerate(unique_labels):
                idx = (labels == domain)
                ax.scatter(
                    embeddings_umap[idx, 0],
                    embeddings_umap[idx, 1],
                    label=domain,
                    color=cmap(i)
                )
            ax.set_xlabel("UMAP Dimension 1")
            ax.set_ylabel("UMAP Dimension 2")
            ax.set_title(title_str)

        # 3D plot
        else:  # embeddings_umap.shape[1] == 3
            ax = fig.add_subplot(111, projection='3d')
            for i, domain in enumerate(unique_labels):
                idx = (labels == domain)
                ax.scatter(
                    embeddings_umap[idx, 0],
                    embeddings_umap[idx, 1],
                    embeddings_umap[idx, 2],
                    label=domain,
                    color=cmap(i),
                    s=20
                )
            ax.set_xlabel("UMAP Dimension 1")
            ax.set_ylabel("UMAP Dimension 2")
            ax.set_zlabel("UMAP Dimension 3")
            ax.set_title(title_str)

        plt.legend()
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300)
        if show:
            plt.show()


"""
# Example usage (assuming torch is imported and cos_umap.py is in PYTHONPATH)

# Suppose we have two domain embeddings: domain A and domain B
embedding_A = torch.rand(10, 64)  # 100 samples, 64-dim embedding
embedding_B = torch.rand(10, 64)  # 120 samples, same 64-dim embedding space

n_neighbors = 15
min_dist = 0.1
n_components = 2
metric = 'cosine'

visualizer = CosineUMAP(n_neighbors=n_neighbors, min_dist=min_dist,
                        n_components=n_components, random_state=42, metric=metric)

emb_2d, labels = visualizer.fit_transform(
    embeddings={"Domain A": embedding_A, "Domain B": embedding_B}
)

# Now emb_2d is an array of shape (220, 2) and labels is an array of length 220.
# We can plot these points, coloring by domain:
import matplotlib.pyplot as plt
fig = plt.figure(figsize=(8, 6))

unique_labels = np.unique(labels)
colors = plt.get_cmap('tab10', len(unique_labels))

if n_components == 1:
    ax = fig.add_subplot(111)
    for i, domain in enumerate(unique_labels):
        idx = (labels == domain)
        ax.scatter(emb_2d[idx, 0], np.zeros_like(emb_2d[idx, 0]), label=domain, color=colors(i))
    ax.set_ylabel("Constant Zero Line")
    ax.set_xlabel("UMAP Dimension 1")

elif n_components == 2:
    ax = fig.add_subplot(111)
    for i, domain in enumerate(unique_labels):
        idx = (labels == domain)
        ax.scatter(emb_2d[idx, 0], emb_2d[idx, 1], label=domain, color=colors(i))
    ax.set_xlabel("UMAP Dimension 1")
    ax.set_ylabel("UMAP Dimension 2")

elif n_components == 3:
    ax = fig.add_subplot(111, projection='3d')
    for i, domain in enumerate(unique_labels):
        idx = (labels == domain)
        ax.scatter(emb_2d[idx, 0], emb_2d[idx, 1], emb_2d[idx, 2], label=domain, s=30, color=colors(i))
    ax.set_xlabel("UMAP Dimension 1")
    ax.set_ylabel("UMAP Dimension 2")
    ax.set_zlabel("UMAP Dimension 3")

plt.title(f"UMAP projection of multi-domain embeddings ({metric})", fontsize=14)
plt.legend()
plt.tight_layout()
plt.show()
# If using MLflow, log the figure:
# import mlflow
# mlflow.log_figure(fig, "multi_domain_umap.png")
"""