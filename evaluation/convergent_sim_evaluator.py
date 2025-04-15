# evaluation/convergent_sim_evaluator.py

import os
import torch
import csv
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

from models.convergent_sim import SimEncoder
from utils.model_utils import ModelUtils
from utils.logger import MLFlowLogger
from evaluation.modules.anomaly_metrics import AnomalyScore, AnomalyDetector
from evaluation.training_eval import eval_latent_alignment

class ConvergentSimEvaluator:
    """
    SimSiam Domain Adaptation evaluator
    Args:
        cfg (dict): config dictionary
        mlflow_logger (MLFlowLogger): MLflow logger(define param in 'main.py')
        run_id (str): MLflow run id
        last_epoch (int): Final epoch to be used for evaluation
        device (torch.device): cuda or cpu
    """
    def __init__(self, cfg: dict, mlflow_logger: MLFlowLogger, run_id: str, last_epoch: int, device: torch.device = None):
        self.cfg = cfg
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # model
        self.encoder = SimEncoder(
            in_dim=self.cfg["mlp"]["in_dim"],
            hidden_dims=self.cfg["mlp"]["in_hidden_dims"],
            latent_dim=self.cfg["mlp"]["latent_dim"],
            dropout=self.cfg["mlp"]["dropout"],
            proj_hidden_dim=self.cfg["sim"]["proj_hidden_dim"],
            proj_out_dim=self.cfg["sim"]["proj_out_dim"],
            pred_hidden_dim=self.cfg["sim"]["pred_hidden_dim"],
            pred_out_dim=self.cfg["sim"]["pred_out_dim"]
        ).to(self.device)

        # utils: model manage, mlflow
        self.model_utils = ModelUtils(self.cfg)
        self.mlflow_logger = mlflow_logger
        self.run_id = run_id
        self.last_epoch = last_epoch

        # params
        self.method = cfg["anomaly"]["method"]
        self.anomaly_score_fn = AnomalyScore(self.cfg)
        self.anomaly_detector = AnomalyDetector(self.cfg, self.anomaly_score_fn)
        self.return_thresholded_preds = cfg["anomaly"].get("return_thresholded_preds", False)
        self.test_n_batch = cfg["test_n_batch"]

    def test_epoch(self, data_loader, epoch):
        ckpt_file = self.model_utils.get_file_name(epoch)
        ckpt_path = self.model_utils.get_file_path(ckpt_file)
        if not Path(ckpt_path).exists():
            raise FileNotFoundError(f"Cannot find checkpoint file: {ckpt_path}")
        
        checkpoint = torch.load(ckpt_path, map_location=self.device)
        self.encoder.load_state_dict(checkpoint['encoder_state_dict'])
        self.encoder.eval()

        all_e_s, all_e_t = [], []
        all_p_s, all_z_s = [], []
        all_p_t, all_z_t = [], []
        all_class_y_s = []
        all_anomaly_y_s = []
        all_class_y_t = []

        # The smaller value between the total length of the current data_loader and eval_n_batch
        if not self.test_n_batch or self.test_n_batch <= 0:
            max_batches = len(data_loader)
        else:
            max_batches = min(len(data_loader), self.test_n_batch)

        pbar = tqdm(enumerate(data_loader), total=max_batches, desc=f"[Test]  [Epoch {epoch}/{self.last_epoch}] | Metric: {self.method}", leave=False)
        with torch.no_grad():
            for batch_idx, data in pbar:
                (x_s, y_s), (x_t, y_t) = data  # y_s, y_t: tensor([class_label, anomaly_label])
                class_y_s = y_s[:,0]  # (B,)
                anomaly_y_s = y_s[:,1]
                class_y_t = y_t[:,0]
                anomaly_y_t = y_t[:,1]

                x_s = x_s.to(self.device)
                x_t = x_t.to(self.device)

                # forward
                e_s, e_t, z_s, p_s, z_t, p_t = self.encoder(x_s, x_t)

                # 250414 검토 필요.
                # stack
                all_e_s.append(e_s.cpu())
                all_e_t.append(e_t.cpu())
                all_p_s.append(p_s.cpu())
                all_z_s.append(z_s.cpu())
                all_p_t.append(p_t.cpu())
                all_z_t.append(z_t.cpu())
                all_class_y_s.append(class_y_s.cpu())
                all_anomaly_y_s.append(anomaly_y_s.cpu())
                all_class_y_t.append(class_y_t.cpu())

                if (batch_idx + 1) >= max_batches:
                    break

        # tensor concat
        all_e_s = torch.cat(all_e_s, dim=0)
        all_e_t = torch.cat(all_e_t, dim=0)
        all_p_s = torch.cat(all_p_s, dim=0)
        all_z_s = torch.cat(all_z_s, dim=0)
        all_p_t = torch.cat(all_p_t, dim=0)
        all_z_t = torch.cat(all_z_t, dim=0)
        all_class_y_s   = torch.cat(all_class_y_s, dim=0).long()
        all_anomaly_y_s = torch.cat(all_anomaly_y_s, dim=0).long()
        all_class_y_t = torch.cat(all_class_y_t, dim=0).long()
        """
        # debug
        unique_class_labels = torch.unique(all_class_labels)
        print(f"[Debug] all_class_labels shape: {all_class_labels.shape}")
        print(f"[Debug] unique class labels: {unique_class_labels}")
        unique_anomaly_labels = torch.unique(all_anomaly_labels)
        print(f"[Debug] all_anomaly_labels shape: {all_anomaly_labels.shape}")
        print(f"[Debug] unique anomaly labels: {unique_anomaly_labels}")
        """
        # Evaluation: AnomalyDetector
        anomaly_dict = {}
        y_pred_opt, anomaly_scores = None, None
        
        ret = self.anomaly_detector.evaluate(p1=all_p_s, z2=all_z_t, p2=all_p_t, z1=all_z_s, 
                                             y_true=all_anomaly_y_s, return_thresholded_preds=self.return_thresholded_preds)
        anomaly_dict, y_pred_opt, anomaly_scores = ret

        # (Optional) csv save
        self.save_anomaly_scores_as_csv(epoch=epoch, anomaly_labels=all_anomaly_y_s, anomaly_scores=anomaly_scores, y_pred_opt=y_pred_opt)

        print(f"[Test]  [Epoch {epoch}/{self.last_epoch}] | Metric: {self.method} | ", anomaly_dict)

        return anomaly_dict, all_e_s, all_e_t, all_z_s, all_z_t, all_class_y_s, all_class_y_t

    def run(self, eval_loader, test_loader):
        """
        Evaluating MLflow run: Load the 0-epoch model & final_epoch model
        """
        if self.mlflow_logger is not None and self.run_id is not None:
            self.mlflow_logger.start_run(self.run_id)
        
        # [Option] For the distribution method, perform fit_distribution() with normal data in advance and use a "normal data-only loader"
        """
        if self.method == "distribution":
            # eval_loader 내 데이터를 '정상'으로 가정
            # 실제론 normal_loader를 별도로 구성하는 편이 바람직
            source_normal, target_normal = self._collect_normal_embeddings(model_init, eval_loader)
            self.anomaly_detector.fit_distribution(source_normal, target_normal)
        """
        # evaluation
        (anomaly_dict_init, all_e_s_init, all_e_t_init, all_z_s_init, all_z_t_init, all_class_y_s_init, all_class_y_t_init) = self.test_epoch(test_loader, epoch=0)
        (anomaly_dict_final, all_e_s_final, all_e_t_final, all_z_s_final, all_z_t_final, all_class_y_s_final, all_class_y_t_final) = self.test_epoch(test_loader, epoch=self.last_epoch)

        # mlflow metrics log
        if self.mlflow_logger:
            self.mlflow_logger.log_metrics({f"init_{k}": v for k, v in anomaly_dict_init.items() if isinstance(v, (float, int))})
            self.mlflow_logger.log_metrics({f"final_{k}": v for k, v in anomaly_dict_final.items() if isinstance(v, (float, int))})

        self._plot_and_log_bar_chart(anomaly_dict_init, anomaly_dict_final)

        eval_latent_alignment(cfg=self.cfg, mlflow_logger=self.mlflow_logger, 
                              source_embeddings=all_e_s_final, target_embeddings=all_e_t_final, 
                              source_labels=all_class_y_s_final, target_labels=all_class_y_t_final, 
                              epoch=self.last_epoch, f_class="encoder")
        eval_latent_alignment(cfg=self.cfg, mlflow_logger=self.mlflow_logger, 
                              source_embeddings=all_z_s_final, target_embeddings=all_z_t_final, 
                              source_labels=all_class_y_s_final, target_labels=all_class_y_t_final, 
                              epoch=self.last_epoch, f_class="projector")

        if self.mlflow_logger:
            self.mlflow_logger.end_run()

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

    def save_anomaly_scores_as_csv(self, epoch: int, anomaly_labels: torch.Tensor, anomaly_scores, y_pred_opt) -> None:
        save_dir = self.model_utils.get_save_dir()
        os.makedirs(f"{save_dir}/metric", exist_ok=True)
        csv_path = f"{save_dir}/metric/anomaly_scores_epoch{epoch}.csv"

        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["index", "anomaly_label", "score", "pred_label"])

            for i, score in enumerate(anomaly_scores):
                a_label = anomaly_labels[i].item()
                pred = int(y_pred_opt[i])
                writer.writerow([i, a_label, float(score), pred])
        # mlflow artifact upload
        if self.mlflow_logger:
            self.mlflow_logger.log_artifact(csv_path, artifact_path="metrics")
            print(f"[save_anomaly_scores_as_csv] CSV saved & logged to MLflow: {csv_path}")