# evaluation/training_eval.py

import os
from utils.model_utils import ModelUtils

from .modules.sample_distance import compute_cross_nn_distance
from .modules.umap import UMAP


def eval_latent_alignment(cfg, mlflow_logger, source_embeddings, target_embeddings, epoch, f_class):
    """
    Evaluate latent space alignment between source and target embeddings for a given epoch.
    Generates a UMAP visualization (using metric distance) of source vs. target embeddings.
    Computes metric nearest-neighbor distance metrics between source and target.
    Logs the resulting plot and metrics to MLflow.
    Args:
        cfg (dict): Configuration dictionary containing UMAP parameters under cfg["umap"].
        mlflow_logger: MLflow logger object with methods log_artifact(path) and log_metrics(dict, step).
        source_embeddings (torch.Tensor): Embeddings from the source domain (shape: [N, dim]).
        target_embeddings (torch.Tensor): Embeddings from the target domain (shape: [M, dim]).
        epoch (int): Current epoch number (used for logging and naming outputs).
    """
    # Extract UMAP configuration parameters
    umap_params = cfg.get("umap", {})
    n_neighbors = umap_params.get("n_neighbors", 15)
    min_dist = umap_params.get("min_dist", 0.1)
    n_components = umap_params.get("n_components", 2)
    random_state = umap_params.get("random_state", 42)
    metric = umap_params.get("metric", None)

    model_utils = ModelUtils(cfg)
    save_dir = model_utils.get_save_dir()
    os.makedirs(f"{save_dir}/umap", exist_ok=True)
    umap_file = f"{save_dir}/umap/umap_epoch{epoch}_{f_class}.png"

    # Create UMAP instance
    cos_umap = UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=n_components,
        random_state=random_state,
        metric=metric
    )

    # Create a dictionary of embeddings for UMAP, labeling each set for visualization
    embeddings_dict = {"source": source_embeddings, "target": target_embeddings}

    # Generate and save the UMAP projection plot using distance
    # UMAP.plot_embeddings will handle fitting UMAP and plotting the two sets of embeddings
    embeddings_2d, label_array = cos_umap.fit_transform(embeddings_dict)

    cos_umap.plot_embeddings(
        embeddings_umap=embeddings_2d,
        labels=label_array,
        save_path=umap_file,
        show=False
    )

    # Log the UMAP plot image to MLflow
    mlflow_logger.log_artifact(umap_file, artifact_path="alignment")

    # Compute cosine nearest-neighbor distance metrics between source and target embeddings
    metrics = compute_cross_nn_distance(source_embeddings, target_embeddings)
    # Log the distance metrics to MLflow (associate them with the current epoch)
    if isinstance(metrics, dict):
        # Log each metric in the dictionary
        mlflow_logger.log_metrics(metrics, step=epoch)
    else:
        # If compute_cross_nn_distance returns a single value or tuple, handle accordingly
        # (This branch can be customized based on the actual return type of compute_cross_nn_distance)
        mlflow_logger.log_metric("cross_nn_distance", metrics, step=epoch)