# evaluation/modules/umap.py

import torch
import numpy as np
import pandas as pd

import umap.umap_ as umap
import matplotlib.pyplot as plt
from matplotlib.patches import PathPatch
from matplotlib.path import Path
from typing import Optional, Tuple


class UMAPPlot:
    """
    Simple UMAP dimensionality reduction and plotting for embeddings.
    """
    def __init__(self, cfg, **umap_kwargs):
        """
        Args:
            n_neighbors: Number of neighbors to consider in UMAP (controls local/global balance).
            min_dist: Minimum distance apart in the low-dimensional space (controls clustering tightness).
            n_components: Dimensionality of reduced embedding (2 for 2D plot, can be 1 or 3 as well).
            random_state: Seed for random number generator (for UMAP initialization and reproducibility).
            metric: Distance metric to use in UMAP. (default: 'None')
            **umap_kwargs: Additional keyword arguments to pass to the umap.UMAP constructor.
            boundary_samples: Number of points to sample on the surface of a high-dimensional hypersphere
            normalize: 
        """
        # UMAP parameters
        umap_params = cfg.get("umap", {})
        self.n_neighbors = umap_params.get("n_neighbors", 15)
        self.min_dist = umap_params.get("min_dist", 0.1)
        self.n_components = umap_params.get("n_components", 2)
        self.random_state = umap_params.get("random_state", 42)
        self.metric = umap_params.get("metric", None)
        self.umap_kwargs = umap_kwargs  # store any additional UMAP parameters provided

        self.boundary_samples = umap_params.get("boundary_samples", None)
        
        self.umap = umap.UMAP(
            n_neighbors=self.n_neighbors,
            min_dist=self.min_dist,
            n_components=self.n_components,
            random_state=self.random_state,
            metric=self.metric,
            **self.umap_kwargs
        )

        # save the UMAP instance for later use (assigned during fit_transform)
        self.reducer = None
        self.norm_min: Optional[np.ndarray] = None
        self.norm_max: Optional[np.ndarray] = None
        self.fix_reducer  = umap_params.get("fix_reducer", False)
        self.normalize = umap_params.get("normalize", True)
        self.update_stats = umap_params.get("update_stats", False)

    def _fit_transform(self, features: np.ndarray) -> np.ndarray:
        if self.fix_reducer:
            if self.reducer is None:
                embedded = self.umap.fit_transform(features)
                self.reducer = self.umap
            else:
                embedded = self.reducer.transform(features)
        else:
            embedded = self.umap.fit_transform(features)
            self.reducer = self.umap
        return embedded
    
    def _minmax_scale(self, arr: np.ndarray) -> np.ndarray:
        update_stats = self.update_stats
        if self.norm_min is None or self.norm_max is None or update_stats:
            self.norm_min = arr.min(axis=0, keepdims=True)
            self.norm_max = arr.max(axis=0, keepdims=True)

        scaled = (arr - self.norm_min) / (self.norm_max - self.norm_min + 1e-8)
        return np.clip(scaled, 0.0, 1.0)
    
    def _export_csv(self, csv_path, embedded, class_labels, anomaly_labels):
        dim_cols = [f"umap_dim{i+1}" for i in range(embedded.shape[1])]
        df = pd.DataFrame(embedded, columns=dim_cols)
        df["class_label"] = class_labels
        df["anomaly_label"] = anomaly_labels
        df.to_csv(csv_path, index=False)

    def plot_umap(self, save_path, features, class_labels, anomaly_labels, 
                  center: Optional[torch.Tensor], radius: Optional[float], boundary_samples: Optional[int], 
                  csv_path: Optional[str] = None):
        """
        Combine embeddings from multiple domains and perform UMAP dimensionality reduction.
        """
        embedded = self._fit_transform(features)  # (N, n_components)

        # export csv
        if csv_path is not None:
            self._export_csv(csv_path, embedded, class_labels, anomaly_labels)

        # normalize the embeddings to fit in the [0, 1] square
        if self.normalize:
            embedded = self._minmax_scale(embedded)
        
        # plot per label
        unique_labels = np.unique(class_labels)
        plt.figure(figsize=(8, 6))
        cmap = plt.get_cmap("tab10", len(unique_labels))

        for i, lbl in enumerate(unique_labels):
            color = cmap(i)
            idx = (class_labels == lbl)
            normal_idx = idx & (anomaly_labels == 0)
            anomaly_idx = idx & (anomaly_labels == 1)

            plt.scatter(
                embedded[normal_idx, 0],
                embedded[normal_idx, 1],
                color=color,
                marker='o',
                alpha=0.7,
                label=f"Class {lbl} - Normal" if np.any(normal_idx) else None
            )
            plt.scatter(
                embedded[anomaly_idx, 0],
                embedded[anomaly_idx, 1],
                color=color,
                marker='x',
                alpha=0.7,
                label=f"Class {lbl} - Anomaly" if np.any(anomaly_idx) else None
            )
        
        # SVDD center & radius
        if center is not None:
            center_np = center.detach().cpu().numpy() if isinstance(center, torch.Tensor) else center
            center_2d = self.reducer.transform(center_np.reshape(1, -1))  # (1, 2)
            cx, cy = center_2d[0, 0], center_2d[0, 1]
            plt.scatter(cx, cy, marker='D', color='red', s=100, label='SVDD Center')  # center

            if radius is not None:  # boundary
                rad_val = radius.item() if hasattr(radius, 'item') else radius
                
                # random sampling the surface of hypersphere
                d = center_np.shape[0]
                rand_vec = np.random.randn(boundary_samples, d)
                rand_vec /= np.linalg.norm(rand_vec, axis=1, keepdims=True)  # unit sphere
                hypersphere_points = center_np + rad_val * rand_vec
                
                boundary_2d = self.reducer.transform(hypersphere_points)  # 2d mapping: (boundary_samples, 2)

                # points are in random order: sort by angle relative to center to connect as outline
                angles = np.arctan2(boundary_2d[:, 1] - cy, boundary_2d[:, 0] - cx)
                sort_idx = np.argsort(angles)
                boundary_2d_sorted = boundary_2d[sort_idx]

                # add Patch: matplotlib Path
                vertices = np.concatenate([
                    boundary_2d_sorted,
                    boundary_2d_sorted[0:1, :]  # Connect first point to close loop
                ], axis=0)
                codes = np.ones(len(vertices), dtype=Path.code_type) * Path.LINETO
                codes[0] = Path.MOVETO
                path = Path(vertices, codes)
                patch = PathPatch(path, facecolor='none', edgecolor='red', linestyle='--', label='SVDD Boundary')
                plt.gca().add_patch(patch)

        plt.title(f"UMAP ({self.metric})")
        plt.xlabel("UMAP Dim 1")
        plt.ylabel("UMAP Dim 2")
        plt.xlim(0, 1) if self.normalize else None
        plt.ylim(0, 1) if self.normalize else None
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()
