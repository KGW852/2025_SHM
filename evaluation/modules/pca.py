# evaluation/modules/pca.py

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import PathPatch
from matplotlib.path import Path
from sklearn.decomposition import PCA
from typing import Optional


class PCAPlot:
    """
    Simple PCA dimensionality reduction and plotting for embeddings.
    """
    def __init__(self, cfg, **pca_kwargs):
        """
        Args:
            n_components: Dimensionality of reduced embedding (2 for 2D plot, etc.).
            random_state: Seed for random number generator (for PCA, if using randomized solver)
            **pca_kwargs: Additional keyword arguments for sklearn.decomposition.PCA.
            boundary_samples: Number of points to sample on the surface of a high-dimensional hypersphere (for plotting SVDD boundary).
        """
        pca_params = cfg.get("pca", {})
        self.n_components = pca_params.get("n_components", 2)
        self.random_state = pca_params.get("random_state", 42)
        self.boundary_samples = pca_params.get("boundary_samples", None)

        # Instantiate PCA with parameters
        self.pca = PCA(
            n_components=self.n_components,
            random_state=self.random_state,
            **pca_kwargs
        )

        self.reducer = None  # will hold fitted PCA model

    def plot_pca(self, save_path: str, features: np.ndarray, class_labels: np.ndarray, anomaly_labels: np.ndarray, 
                 center: Optional[torch.Tensor] = None, radius: Optional[float] = None, boundary_samples: Optional[int] = None):
        """
        Perform PCA dimensionality reduction and 2D scatter plot of embeddings.
        Args:
            save_path: Path to save the PCA plot (e.g., "results/pca_plot.png").
            features: (N, D) feature matrix to reduce and plot.
            class_labels: (N,) array of class labels for coloring.
            anomaly_labels: (N,) array of anomaly labels (0: normal, 1: anomaly).
            center: (D,) center coordinate (e.g., SVDD center in the original feature space).
            radius: Float, radius of the boundary in the original feature space.
            boundary_samples: Integer, number of boundary points to sample for drawing the hypersphere.
        """
        # Fit PCA and transform features
        embedded = self.pca.fit_transform(features)  # (N, n_components)
        self.reducer = self.pca  # Store for possible reuse

        # 2D plotting
        if self.n_components < 2:
            raise ValueError("PCAPlot is set up for 2D visualization. Please use n_components>=2.")

        plt.figure(figsize=(8, 6))
        unique_labels = np.unique(class_labels)
        cmap = plt.get_cmap("tab10", len(unique_labels))

        # Plot per class label with normal/anomaly distinction
        for i, lbl in enumerate(unique_labels):
            color = cmap(i)
            idx = (class_labels == lbl)
            normal_idx = idx & (anomaly_labels == 0)
            anomaly_idx = idx & (anomaly_labels == 1)

            # Normal samples
            if np.any(normal_idx):
                plt.scatter(
                    embedded[normal_idx, 0],
                    embedded[normal_idx, 1],
                    color=color,
                    marker='o',
                    alpha=0.7,
                    label=f"Class {lbl} - Normal"
                )

            # Anomaly samples
            if np.any(anomaly_idx):
                plt.scatter(
                    embedded[anomaly_idx, 0],
                    embedded[anomaly_idx, 1],
                    color=color,
                    marker='x',
                    alpha=0.7,
                    label=f"Class {lbl} - Anomaly"
                )

        # SVDD center & boundary (if provided)
        if center is not None:
            center_np = (
                center.detach().cpu().numpy()
                if isinstance(center, torch.Tensor)
                else center
            )
            # PCA transform for center
            center_2d = self.reducer.transform(center_np.reshape(1, -1))  # (1, 2)
            cx, cy = center_2d[0, 0], center_2d[0, 1]
            plt.scatter(cx, cy, marker='D', color='red', s=100, label='SVDD Center')

            if radius is not None and boundary_samples is not None:
                rad_val = radius.item() if hasattr(radius, 'item') else radius

                # Randomly sample the surface of a D-dim hypersphere
                d = center_np.shape[0]
                rand_vec = np.random.randn(boundary_samples, d)
                rand_vec /= np.linalg.norm(rand_vec, axis=1, keepdims=True)  # Project to unit sphere
                hypersphere_points = center_np + rad_val * rand_vec

                # Transform boundary points into PCA space
                boundary_2d = self.reducer.transform(hypersphere_points)  # (boundary_samples, 2)

                # Sort boundary points by angle for continuous boundary
                angles = np.arctan2(boundary_2d[:, 1] - cy, boundary_2d[:, 0] - cx)
                sort_idx = np.argsort(angles)
                boundary_2d_sorted = boundary_2d[sort_idx]

                # Construct closed path
                vertices = np.concatenate([
                    boundary_2d_sorted,
                    boundary_2d_sorted[0:1, :],
                ], axis=0)
                codes = np.ones(len(vertices), dtype=Path.code_type) * Path.LINETO
                codes[0] = Path.MOVETO
                path = Path(vertices, codes)
                patch = PathPatch(
                    path,
                    facecolor='none',
                    edgecolor='red',
                    linestyle='--',
                    label='SVDD Boundary'
                )
                plt.gca().add_patch(patch)

        plt.title("PCA")
        plt.xlabel("PCA Dim 1")
        plt.ylabel("PCA Dim 2")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()
