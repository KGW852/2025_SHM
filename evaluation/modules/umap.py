# evaluation/modules/umap.py

import os
import numpy as np
import torch
import umap
import matplotlib.pyplot as plt

from typing import Dict, List, Tuple, Union, Optional

from utils.model_utils import ModelUtils

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

        # Save the UMAP instance for later use (assigned during fit_transform)
        self.reducer = None
        self.embedding_ = None
        self.labels_ = None
    

    def fit_transform(self, 
                      embeddings: Union[Dict[str, torch.Tensor], List[torch.Tensor]], 
                      labels: Optional[List[str]] = None
                     ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Combine embeddings from multiple domains and perform UMAP dimensionality reduction.
        """
        if isinstance(embeddings, dict):
            data_items = list(embeddings.items())
        elif isinstance(embeddings, list):
            if labels is None or len(labels) != len(embeddings):
                raise ValueError("When providing embeddings as a list, "
                                 "a list of labels of the same length must be provided.")
            data_items = list(zip(labels, embeddings))
        else:
            raise ValueError("Embeddings must be a dict {label: tensor} or a list of tensors + labels.")

        combined_data_list: List[np.ndarray] = []
        combined_labels: List[str] = []
        expected_dim: Optional[int] = None
        
        for domain_label, tensor in data_items:
            if isinstance(tensor, torch.Tensor):
                tensor_np = tensor.detach().cpu().numpy()
            elif isinstance(tensor, np.ndarray):
                tensor_np = tensor
            else:
                try:  # Try to convert other sequence types to numpy
                    tensor_np = np.array(tensor)
                except Exception as e:
                    raise ValueError(f"Embedding data for label '{domain_label}' is not a tensor/ndarray and cannot be converted: {e}")
            
            # Validate shape
            if tensor_np.ndim != 2:
                raise ValueError(f"Embedding for label '{domain_label}' must be 2D, got {tensor_np.shape}.")
            
            if expected_dim is None:
                expected_dim = tensor_np.shape[1]
            elif tensor_np.shape[1] != expected_dim:
                raise ValueError(f"Dimension mismatch: domain '{domain_label}' has latent_dim {tensor_np.shape[1]}, expected {expected_dim}.")
            
            # Append data and labels
            combined_data_list.append(tensor_np)
            # Repeat the domain label for each sample in this tensor
            num_samples = tensor_np.shape[0]
            combined_labels.extend([domain_label] * num_samples)
        
        if len(combined_data_list) == 0:
            raise ValueError("No embeddings provided for UMAP visualization.")
        
        combined_data = np.vstack(combined_data_list)
        
        # Configure and fit UMAP
        self.reducer = umap.UMAP(n_neighbors=self.n_neighbors,
                                 min_dist=self.min_dist,
                                 n_components=self.n_components,
                                 metric=self.metric,
                                 random_state=self.random_state,
                                 **self.umap_kwargs)
        
        embeddings_2d = self.reducer.fit_transform(combined_data)
        label_array = np.array(combined_labels)
        
        # Store for later use
        self.embedding_ = embeddings_2d
        self.labels_ = label_array
        return embeddings_2d, label_array

    def plot_embeddings(self, embeddings_umap: Optional[np.ndarray] = None, labels: Optional[np.ndarray] = None,
                        src_center: Optional[np.ndarray] = None, src_radian: Optional[float] = None,
                        tgt_center: Optional[np.ndarray] = None, tgt_radian: Optional[float] = None,
                        save_path: Optional[str] = None, show: bool = True) -> None:
        """
        Plot the UMAP-transformed embeddings with color-coded domain labels. Supports 1D, 2D, or 3D embeddings.
        Optionally, also plot circles for src/tgt centers and their radii (SVDD).
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

            # SVDD center & radius: Transform src_center, tgt_center to UMAP 2D for visualization
            if self.reducer is not None:
                if src_center is not None:
                    center_s_np = src_center.reshape(1, -1)
                    center_s_2d = self.reducer.transform(center_s_np)  # shape: (1, 2)
                    ax.scatter(center_s_2d[0, 0], center_s_2d[0, 1], marker='x', s=100, color='red', label='src_center')
                    if src_radian is not None:  # display radius in 2D space
                        circle_s = plt.Circle((center_s_2d[0, 0], center_s_2d[0, 1]), radius=src_radian, color='red', fill=False, linestyle='--', label='src_radian')
                        ax.add_patch(circle_s)

                if tgt_center is not None:
                    center_t_np = tgt_center.reshape(1, -1)
                    center_t_2d = self.reducer.transform(center_t_np)
                    ax.scatter(center_t_2d[0, 0], center_t_2d[0, 1], marker='x', s=100, color='blue', label='tgt_center') 
                    if tgt_radian is not None:
                        circle_t = plt.Circle((center_t_2d[0, 0], center_t_2d[0, 1]), radius=tgt_radian, color='blue', fill=False, linestyle='--', label='tgt_radian')
                        ax.add_patch(circle_t)

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


def plot_latent_alignment(cfg, mlflow_logger, src_embed, tgt_embed, src_lbl, tgt_lbl, epoch, f_name, src_center=None, src_radian=None, tgt_center=None, tgt_radian=None):
    """
    Evaluate latent space alignment between source and target embeddings for a given epoch, grouped by class labels. Optionally plot SVDD center & radius.
    """
    # UMAP parameters
    umap_params = cfg.get("umap", {})
    n_neighbors = umap_params.get("n_neighbors", 15)
    min_dist = umap_params.get("min_dist", 0.1)
    n_components = umap_params.get("n_components", 2)
    random_state = umap_params.get("random_state", 42)
    metric = umap_params.get("metric", None)

    model_utils = ModelUtils(cfg)
    save_dir = model_utils.get_save_dir()
    os.makedirs(f"{save_dir}/umap", exist_ok=True)
    umap_file = f"{save_dir}/umap/umap_epoch{epoch}_{f_name}.png"

    # UMAP instance
    cos_umap = UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=n_components,
        random_state=random_state,
        metric=metric
    )

    # label, embeddings dict
    unique_labels = torch.unique(torch.cat([src_lbl, tgt_lbl], dim=0))
    embeddings_dict = {}
    for c in unique_labels:
        mask_s = (src_lbl == c)
        if mask_s.any():
            embeddings_dict[f"src_class_{c.item()}"] = src_embed[mask_s]
        mask_t = (tgt_lbl == c)
        if mask_t.any():
            embeddings_dict[f"tgt_class_{c.item()}"] = tgt_embed[mask_t]

    # UMAP.plot_embeddings
    embeddings_2d, label_array = cos_umap.fit_transform(embeddings_dict)
    cos_umap.plot_embeddings(embeddings_umap=embeddings_2d, labels=label_array, save_path=umap_file, show=False, 
                             src_center=src_center, src_radian=src_radian, tgt_center=tgt_center, tgt_radian=tgt_radian)

    # Log the UMAP plot image to MLflow
    mlflow_logger.log_artifact(umap_file, artifact_path="alignment")
