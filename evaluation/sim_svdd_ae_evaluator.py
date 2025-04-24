# evaluation/sim_svdd_ae_evaluator.py

import os
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm

from models.sim_svdd_ae import SimSVDDAE
from utils.model_utils import ModelUtils
from utils.logger import MLFlowLogger
from utils.datalist_utils import remove_duplicates
from evaluation.modules.anomaly_metrics import AnomalyScore, AnomalyMetric
from evaluation.modules.umap import UMAPPlot
from evaluation.modules.pca import PCAPlot


class SimSVDDAEEvaluator:
    """
    SimSiam Domain Adaptation evaluator
    Args:
        cfg (dict): config dictionary
        mlflow_logger (MLFlowLogger): MLflow logger(define param in 'main.py')
        device (torch.device): cuda or cpu
    """
    def __init__(self, cfg: dict, mlflow_logger: MLFlowLogger, device: torch.device = None):
        self.cfg = cfg
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # model
        self.model = SimSVDDAE(
            enc_in_dim=cfg["mlp"]["in_dim"],
            enc_hidden_dims=cfg["mlp"]["enc_hidden_dims"],
            enc_latent_dim=cfg["mlp"]["enc_latent_dim"],
            dec_latent_dim=cfg["mlp"]["dec_latent_dim"],
            dec_hidden_dims=cfg["mlp"]["dec_hidden_dims"],
            dec_out_channels=cfg["mlp"]["out_channels"],
            dec_out_seq_len=cfg["mlp"]["out_seq_len"],
            ae_dropout=cfg["mlp"]["dropout"],
            ae_use_batchnorm=cfg["mlp"]["use_batch_norm"],

            proj_hidden_dim=cfg["sim"]["proj_hidden_dim"],
            proj_out_dim=cfg["sim"]["proj_out_dim"],
            pred_hidden_dim=cfg["sim"]["pred_hidden_dim"],
            pred_out_dim=cfg["sim"]["pred_out_dim"],

            svdd_in_dim=cfg["svdd"]["in_dim"],
            svdd_hidden_dims=cfg["svdd"]["hidden_dims"],
            svdd_latent_dim=cfg["svdd"]["latent_dim"],
            svdd_dropout=cfg["svdd"]["dropout"],
            svdd_use_batchnorm=cfg["svdd"]["use_batch_norm"]
        ).to(self.device)

        # run param
        self.eval_run_epoch = cfg["eval"]["epoch"]
        self.test_run_epoch = cfg["test"]["epoch"]

        # utils: model manage, mlflow
        self.model_utils = ModelUtils(self.cfg)
        self.mlflow_logger = mlflow_logger

        # run_name: model_name
        self.run_name = self.model_utils.get_model_name()

        # metrics modules
        self.anomaly_score = AnomalyScore(self.cfg)
        self.umap = UMAPPlot(self.cfg)
        self.pca = PCAPlot(self.cfg)
        self.method = cfg["anomaly"]["method"]

        # center, radius
        self.center = None
        self.radius = None

    def load_checkpoint(self, epoch: int):
        file_name = self.model_utils.get_file_name(epoch)
        ckpt_path = self.model_utils.get_file_path(file_name)
        if not Path(ckpt_path).exists():
            raise FileNotFoundError(f"Cannot find checkpoint file: {ckpt_path}")

        checkpoint = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])

        self.center = self.model.svdd.center.detach().clone()
        self.radius = self.model.svdd.radius.detach().clone()

    def test_epoch(self, data_loader, epoch):
        self.load_checkpoint(epoch)
        self.model.eval()

        src_results, tgt_results = [], []

        pbar = tqdm(enumerate(data_loader), total=len(data_loader), desc=f"[Test]  [Epoch {epoch}] | Metric: {self.method}", leave=False)
        with torch.no_grad():
            for batch_idx, data in pbar:
                (src_data, src_label, src_path), (tgt_data, tgt_label, tgt_path) = data  # src_label, tgt_label: tensor([class_label, anomaly_label])
                x_s = src_data.to(self.device)
                x_t = tgt_data.to(self.device)

                # forward
                (e_s, e_t, z_s, p_s, z_t, p_t, feat_s, feat_t, dist_s, dist_t, x_s_recon, x_t_recon) = self.model(x_s, x_t)

                batch_size = len(src_path)
                for i in range(batch_size):
                    class_label_src = src_label[i][0].item()
                    anomaly_label_src = src_label[i][1].item()
                    class_label_tgt = tgt_label[i][0].item()
                    anomaly_label_tgt = tgt_label[i][1].item()

                    # results
                    src_results.append({
                        "file_name"     : src_path[i],
                        "x"             : x_s[i].cpu().numpy(),
                        "encoder"       : e_s[i].cpu().numpy(),
                        "projector"     : z_s[i].cpu().numpy(),
                        "feature"       : feat_s[i].cpu().numpy(),
                        "distance"      : dist_s[i].cpu().numpy(),
                        "x_recon"       : x_s_recon[i].cpu().numpy(),
                        "class_label"   : class_label_src,
                        "anomaly_label" : anomaly_label_src,
                        "domain"        : "src"
                    })
                    tgt_results.append({
                        "file_name"     : tgt_path[i],
                        "x"             : x_t[i].cpu().numpy(),
                        "encoder"       : e_t[i].cpu().numpy(),
                        "projector"     : z_t[i].cpu().numpy(),
                        "feature"       : feat_t[i].cpu().numpy(),
                        "distance"      : dist_t[i].cpu().numpy(),
                        "x_recon"       : x_t_recon[i].cpu().numpy(),
                        "class_label"   : class_label_tgt,
                        "anomaly_label" : anomaly_label_tgt,
                        "domain"        : "tgt"
                    })

                    src_results = remove_duplicates(src_results, key_name="file_name")
                    tgt_results = remove_duplicates(tgt_results, key_name="file_name")
                    combined_results = src_results + tgt_results

        return combined_results

    def run(self, eval_loader, test_loader):
        """
        Evaluating MLflow run: Load the epoch model
        """
        if self.mlflow_logger is not None:
            run_id_path = self.model_utils.get_file_path(f"{self.run_name}.json")
            run_id, exp_id, exp_name = self.mlflow_logger.load_run_id(run_id_path)
            self.mlflow_logger.resume_run(run_id)

        # run epochs extract
        test_run_epochs = self.model_utils.get_run_epochs(self.test_run_epoch)

        # evaluation: test_loader
        for epoch in test_run_epochs:
            test_results = self.test_epoch(test_loader, epoch)

            # evaluation: extract dict
            file_names, y_true, y_scores = [], [], []
            enc_list, feat_list = [], []
            class_labels, anomaly_labels = [], []

            for res in test_results:
                file_name = res["file_name"]
                anomaly_label = res["anomaly_label"]
                
                x = torch.tensor(res["x"]).unsqueeze(0).to(self.device)
                x_recon = torch.tensor(res["x_recon"]).unsqueeze(0).to(self.device)
                score_tensor = self.anomaly_score.anomaly_score(x=x, x_recon=x_recon)
                #feat = torch.tensor(res["feature"]).unsqueeze(0).to(self.device)  # (1, latent_dim)
                #score_tensor = self.anomaly_score.anomaly_score(feature=feat, center=self.center)

                file_names.append(file_name)
                y_true.append(anomaly_label)
                y_scores.append(score_tensor.item())

                enc_list.append(res["encoder"])  # (latent_dim,)
                feat_list.append(res["feature"])
                class_labels.append(res["class_label"])
                anomaly_labels.append(res["anomaly_label"])

            # evaluation: UMAP, PCA
            save_dir = self.model_utils.get_save_dir()
            os.makedirs(f"{save_dir}/umap", exist_ok=True)
            enc_umap_path = f"{save_dir}/umap/umap_encoder_epoch{epoch}.png"
            feat_umap_path = f"{save_dir}/umap/umap_feature_epoch{epoch}.png"

            os.makedirs(f"{save_dir}/pca", exist_ok=True)
            feat_pca_path = f"{save_dir}/pca/pca_feature_epoch{epoch}.png"
            
            enc_np = np.stack(enc_list, axis=0)  # (N, latent_dim)
            feat_np = np.stack(feat_list, axis=0)
            class_np = np.array(class_labels)  # (N,)
            anomaly_np = np.array(anomaly_labels)

            self.umap.plot_umap(
                save_path=enc_umap_path,
                features=enc_np,
                class_labels=class_np,
                anomaly_labels=anomaly_np,
                center=None,
                radius=None,
                boundary_samples=None
            )
            self.umap.plot_umap(
                save_path=feat_umap_path,
                features=feat_np,
                class_labels=class_np,
                anomaly_labels=anomaly_np,
                center=self.center,
                radius=self.radius,
                boundary_samples=self.umap.boundary_samples
            )
            self.pca.plot_pca(
                save_path=feat_pca_path,
                features=feat_np,
                class_labels=class_np,
                anomaly_labels=anomaly_np,
                center=self.center,
                radius=self.radius,
                boundary_samples=self.pca.boundary_samples
            )

            # evaluation: AnomalyMetric
            os.makedirs(f"{save_dir}/metric", exist_ok=True)
            anomaly_data_path = f"{save_dir}/metric/anomaly_scores_epoch{epoch}_{self.method}.csv"
            anomaly_metric_path = f"{save_dir}/metric/anomaly_metric_epoch{epoch}_{self.method}.csv"

            anomaly_metric = AnomalyMetric(cfg=self.cfg, file_name=file_names, y_true=y_true, y_score=y_scores)
            anomaly_dict = anomaly_metric.calc_metric()
            print(f"[Test]  [Epoch {epoch}] | Metric: {self.method} | ", anomaly_dict)

            anomaly_metric.save_anomaly_scores_as_csv(data_csv_path=anomaly_data_path, metric_csv_path=anomaly_metric_path)

            # save and mlflow artifact upload
            if self.mlflow_logger:
                self.mlflow_logger.log_artifact(anomaly_data_path, artifact_path="metrics")
                print(f"[Info] AnomalyMetrics saved & logged to MLflow: {anomaly_data_path}")
                self.mlflow_logger.log_artifact(anomaly_metric_path, artifact_path="metrics")
                print(f"[Info] AnomalyMetrics saved & logged to MLflow: {anomaly_metric_path}")
                self.mlflow_logger.log_artifact(enc_umap_path, artifact_path="umap")
                print(f"[Info] UMAP saved & logged to MLflow: {enc_umap_path}")
                self.mlflow_logger.log_artifact(feat_umap_path, artifact_path="umap")
                print(f"[Info] UMAP saved & logged to MLflow: {feat_umap_path}")
                self.mlflow_logger.log_artifact(feat_pca_path, artifact_path="pca")
                print(f"[Info] PCA saved & logged to MLflow: {feat_pca_path}")

        if self.mlflow_logger:
            self.mlflow_logger.end_run()
"""
    def _plot_and_log_bar_chart(self, metrics_init: dict, metrics_final: dict):
        metric_keys = ["accuracy", "precision", "recall", "f1", "auc"]
        init_vals = [metrics_init.get(k, 0.0) for k in metric_keys]
        final_vals = [metrics_final.get(k, 0.0) for k in metric_keys]

        plt.figure(figsize=(8, 5))
        x = range(len(metric_keys))
        width = 0.35

        plt.bar([i - width/2 for i in x], init_vals, width=width, label="Init (0 epoch)", color='blue')
        plt.bar([i + width/2 for i in x], final_vals, width=width, label=f"Final ({self.last_epoch} epoch)", color='orange')

        plt.xticks(list(x), metric_keys)
        plt.ylim(0, 1.05)
        plt.title("Init vs Final Epoch Metric Comparison")
        plt.legend()
        plt.tight_layout()

        # save fig
        save_dir = self.model_utils.get_save_dir()
        os.makedirs(f"{save_dir}/metric", exist_ok=True)
        fig_path = f"{save_dir}/metric/metric_init and epoch{self.last_epoch}.png"
        plt.savefig(fig_path, dpi=100)
        plt.close()
        # mlflow artifact upload
        if self.mlflow_logger is not None:
            self.mlflow_logger.log_artifact(fig_path, artifact_path="metrics")
            print(f"Bar chart saved & logged to MLflow: {fig_path}")
"""