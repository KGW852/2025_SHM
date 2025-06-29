# evaluation/resnet_ae_evaluator.py

import os
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm

from models.resnet_ae import ResNetAE
from utils.model_utils import ModelUtils
from utils.logger import MLFlowLogger
from utils.datalist_utils import remove_duplicates
from evaluation.modules.anomaly_metrics import AnomalyScore, AnomalyMetric
from evaluation.modules.umap import UMAPPlot
from evaluation.modules.pca import PCAPlot
from evaluation.modules.all_scores import ExtractScore


class ResNetAEEvaluator:
    """
    Evaluator for ResNet AutoEncoder (ResNetAE) model.
    Args:
        cfg (dict): Configuration dictionary
        mlflow_logger (MLFlowLogger): MLflow logger instance
        device (torch.device): Device to run the model on (cuda or cpu)
    """
    def __init__(self, cfg: dict, mlflow_logger: MLFlowLogger, device: torch.device = None):
        self.cfg = cfg
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Model initialization
        self.model = ResNetAE(
            channels=cfg["resnet_ae"].get("channels", 3),
            height=cfg["resnet_ae"].get("height"),
            width=cfg["resnet_ae"].get("width"),
            enc_freeze=cfg["resnet_ae"].get("enc_freeze", True),
            dec_latent_dim=cfg["resnet_ae"].get("dec_latent_dim", 2048),
            dec_hidden_dims=cfg["resnet_ae"].get("dec_hidden_dims"),
            dropout=cfg["resnet_ae"].get("dropout", 0.0),
            use_batchnorm=cfg["resnet_ae"].get("use_batch_norm", False)
        ).to(self.device)

        # Run parameters
        self.eval_run_epoch = cfg["eval"]["epoch"]
        self.test_run_epoch = cfg["test"]["epoch"]

        # Utilities
        self.model_utils = ModelUtils(self.cfg)
        self.mlflow_logger = mlflow_logger
        self.run_name = self.model_utils.get_model_name()

        # Metrics modules
        self.anomaly_score = AnomalyScore(self.cfg)
        self.umap = UMAPPlot(self.cfg)
        self.pca = PCAPlot(self.cfg)
        self.extract_score = ExtractScore(self.cfg)
        self.method = cfg["anomaly"]["method"]

    def load_checkpoint(self, epoch: int):
        """Load model checkpoint for a specified epoch."""
        file_name = self.model_utils.get_file_name(epoch)
        ckpt_path = self.model_utils.get_file_path(file_name)
        if not Path(ckpt_path).exists():
            raise FileNotFoundError(f"Cannot find checkpoint file: {ckpt_path}")

        checkpoint = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])

    def test_epoch(self, data_loader, epoch):
        """Run a test epoch on the data loader and collect results."""
        self.load_checkpoint(epoch)
        self.model.eval()

        src_results, tgt_results = [], []

        pbar = tqdm(enumerate(data_loader), total=len(data_loader), desc=f"[Test]  [Epoch {epoch}] | Metric: {self.method}", leave=False)
        with torch.no_grad():
            for batch_idx, data in pbar:
                (src_data, src_label, src_path), (tgt_data, tgt_label, tgt_path) = data
                x_s = src_data.to(self.device)
                x_t = tgt_data.to(self.device)

                # Forward pass
                (e_s, e_t, x_s_recon, x_t_recon) = self.model(x_s, x_t)

                batch_size = len(src_path)
                for i in range(batch_size):
                    class_label_src = src_label[i][0].item()
                    anomaly_label_src = src_label[i][1].item()
                    class_label_tgt = tgt_label[i][0].item()
                    anomaly_label_tgt = tgt_label[i][1].item()

                    # Collect results
                    src_results.append({
                        "file_name": src_path[i],
                        "x": x_s[i].cpu().numpy(),
                        "encoder": e_s[i].cpu().numpy(),
                        "x_recon": x_s_recon[i].cpu().numpy(),
                        "class_label": class_label_src,
                        "anomaly_label": anomaly_label_src,
                        "domain": "src"
                    })
                    tgt_results.append({
                        "file_name": tgt_path[i],
                        "x": x_t[i].cpu().numpy(),
                        "encoder": e_t[i].cpu().numpy(),
                        "x_recon": x_t_recon[i].cpu().numpy(),
                        "class_label": class_label_tgt,
                        "anomaly_label": anomaly_label_tgt,
                        "domain": "tgt"
                    })

                    src_results = remove_duplicates(src_results, key_name="file_name")
                    tgt_results = remove_duplicates(tgt_results, key_name="file_name")
                    combined_results = src_results + tgt_results

        return combined_results, src_results

    def run(self, eval_loader, test_loader):
        """Run evaluation and testing for specified epochs."""
        if self.mlflow_logger is not None:
            run_id_path = self.model_utils.get_file_path(f"{self.run_name}.json")
            run_id, exp_id, exp_name = self.mlflow_logger.load_run_id(run_id_path)
            self.mlflow_logger.resume_run(run_id)

        test_run_epochs = self.model_utils.get_run_epochs(self.test_run_epoch)

        for epoch in test_run_epochs:
            test_results, test_src_results = self.test_epoch(test_loader, epoch)

            # Latent space visualization
            enc_list = [res["encoder"] for res in test_results]
            class_labels = [res["class_label"] for res in test_results]
            anomaly_labels = [res["anomaly_label"] for res in test_results]

            save_dir = self.model_utils.get_save_dir()
            os.makedirs(f"{save_dir}/umap", exist_ok=True)
            enc_umap_path = f"{save_dir}/umap/umap_s2(18)_encoder_epoch{epoch}.png"
            enc_umap_data_path = f"{save_dir}/umap/umap_s2(18)_encoder_epoch{epoch}.csv"

            os.makedirs(f"{save_dir}/pca", exist_ok=True)
            enc_pca_path = f"{save_dir}/pca/pca_s2(18)_encoder_epoch{epoch}.png"

            enc_np = np.stack(enc_list, axis=0)
            class_np = np.array(class_labels)
            anomaly_np = np.array(anomaly_labels)

            self.umap.plot_umap(
                save_path=enc_umap_path,
                csv_path = enc_umap_data_path,
                features=enc_np,
                class_labels=class_np,
                anomaly_labels=anomaly_np,
                center=None,
                radius=None,
                boundary_samples=None
            )
            self.pca.plot_pca(
                save_path=enc_pca_path,
                features=enc_np,
                class_labels=class_np,
                anomaly_labels=anomaly_np,
                center=None,
                radius=None,
                boundary_samples=None
            )

            # Anomaly detection performance evaluation
            file_names, y_true, y_scores = [], [], []
            for src_res in test_src_results:
                file_name = src_res["file_name"]
                anomaly_label = src_res["anomaly_label"]

                x = torch.tensor(src_res["x"]).unsqueeze(0).to(self.device)
                x_recon = torch.tensor(src_res["x_recon"]).unsqueeze(0).to(self.device)
                score_tensor = self.anomaly_score.anomaly_score(x_recon=x_recon, x=x)

                file_names.append(file_name)
                y_true.append(anomaly_label)
                y_scores.append(score_tensor.item())

            os.makedirs(f"{save_dir}/metric", exist_ok=True)
            anomaly_data_path = f"{save_dir}/metric/anomaly_s2(18)_scores_epoch{epoch}_{self.method}.csv"
            anomaly_metric_path = f"{save_dir}/metric/anomaly_s2(18)_metric_epoch{epoch}_{self.method}.csv"

            # Extract all scores
            os.makedirs(f"{save_dir}/scores", exist_ok=True)
            recon_scores_path = f"{save_dir}/scores/s2(18)_scores_epoch{epoch}_recon.csv"

            # Results
            anomaly_metric = AnomalyMetric(cfg=self.cfg, file_name=file_names, y_true=y_true, y_score=y_scores)
            anomaly_dict = anomaly_metric.calc_metric()
            print(f"[Test]  [Epoch {epoch}] | Metric: {self.method} | ", anomaly_dict)

            anomaly_metric.save_anomaly_scores_as_csv(data_csv_path=anomaly_data_path, metric_csv_path=anomaly_metric_path)

            self.extract_score.recon_scores(csv_path=recon_scores_path, results_list=test_results)

            if self.mlflow_logger:
                self.mlflow_logger.log_artifact(enc_umap_path, artifact_path="umap")
                self.mlflow_logger.log_artifact(enc_umap_data_path, artifact_path="umap")
                self.mlflow_logger.log_artifact(enc_pca_path, artifact_path="pca")
                self.mlflow_logger.log_artifact(anomaly_data_path, artifact_path="metrics")
                self.mlflow_logger.log_artifact(anomaly_metric_path, artifact_path="metrics")
                self.mlflow_logger.log_artifact(recon_scores_path, artifact_path="scores")

        if self.mlflow_logger:
            self.mlflow_logger.end_run()