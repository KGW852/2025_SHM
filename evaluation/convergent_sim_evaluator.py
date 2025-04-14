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

    def test_epoch(self, data_loader, epoch):
        ckpt_file = self.model_utils.get_file_name(epoch)
        ckpt_path = self.model_utils.get_file_path(ckpt_file)
        if not Path(ckpt_path).exists():
            raise FileNotFoundError(f"Cannot find checkpoint file: {ckpt_path}")
        
        checkpoint = torch.load(ckpt_path, map_location=self.device)
        self.encoder.load_state_dict(checkpoint['encoder_state_dict'])
        self.encoder.eval()

        all_p_s, all_z_s = [], []
        all_p_t, all_z_t = [], []
        all_class_labels = []
        all_anomaly_labels = []

        pbar = tqdm(data_loader, desc=f"[Test]  [Epoch {epoch}/{self.last_epoch}] | Metric: {self.method}", leave=False)
        with torch.no_grad():
            for (x_s, y_s), (x_t, y_t) in pbar:  # y_s, y_t: tensor([class_label, anomaly_label])
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
                all_p_s.append(p_s.cpu())
                all_z_s.append(z_s.cpu())
                all_p_t.append(p_t.cpu())
                all_z_t.append(z_t.cpu())
                all_class_labels.append(class_y_s.cpu())
                all_anomaly_labels.append(anomaly_y_s.cpu())

        # tensor concat
        all_p_s = torch.cat(all_p_s, dim=0)
        all_z_s = torch.cat(all_z_s, dim=0)
        all_p_t = torch.cat(all_p_t, dim=0)
        all_z_t = torch.cat(all_z_t, dim=0)
        all_class_labels   = torch.cat(all_class_labels, dim=0).long()
        all_anomaly_labels = torch.cat(all_anomaly_labels, dim=0).long()
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
        results_dict = {}
        y_pred_opt, anomaly_scores = None, None
        
        ret = self.anomaly_detector.evaluate(p1=all_p_s, z2=all_z_t, p2=all_p_t, z1=all_z_s, 
                                             y_true=all_anomaly_labels, return_thresholded_preds=self.return_thresholded_preds)
        results_dict, y_pred_opt, anomaly_scores = ret

        # (Optional) csv save
        self.save_anomaly_scores_as_csv(epoch=epoch, anomaly_labels=all_anomaly_labels, anomaly_scores=anomaly_scores, y_pred_opt=y_pred_opt)

        print(f"[Test]  [Epoch {epoch}/{self.last_epoch}] | Metric: {self.method} | ", results_dict)

        return results_dict

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
        results_init = self.test_epoch(test_loader, epoch=0)
        results_final = self.test_epoch(test_loader, epoch=self.last_epoch)

        # mlflow metrics log
        if self.mlflow_logger is not None:
            # init
            init_metrics = {}
            for key, val in results_init.items():
                if isinstance(val, (float, int)):
                    init_metrics[f"init_{key}"] = val
            self.mlflow_logger.log_metrics(init_metrics)
            # final
            final_metrics = {}
            for key, val in results_final.items():
                if isinstance(val, (float, int)):
                    final_metrics[f"final_{key}"] = val
            self.mlflow_logger.log_metrics(final_metrics)

        self._plot_and_log_bar_chart(results_init, results_final)
        self.mlflow_logger.end_run()

    def _plot_and_log_bar_chart(self, metrics_init: dict, metrics_final: dict):
        metric_keys = ["F1", "Accuracy", "Precision", "Recall", "AUC"]
        init_values = [metrics_init.get(k, 0) for k in metric_keys]
        final_values = [metrics_final.get(k, 0) for k in metric_keys]

        x_labels = metric_keys
        x = range(len(x_labels))  # x index

        plt.figure(figsize=(8, 5))
        width = 0.35

        # init bar
        plt.bar([i - width/2 for i in x], init_values, width=width, label="Init Epoch", color='blue')
        # final bar
        plt.bar([i + width/2 for i in x], final_values, width=width, label=f"Final Epoch({self.last_epoch})", color='orange')

        plt.xticks(x, x_labels)
        plt.ylabel("Metric Value")
        plt.ylim(0, 1.05)
        plt.title("Comparison of Init vs Final Epoch Metrics")
        plt.legend()
        plt.tight_layout()

        # save fig
        save_dir = self.model_utils.get_save_dir()
        os.makedirs(f"{save_dir}/metric", exist_ok=True)
        metrics_file = f"{save_dir}/metric/metrics.png"
        plt.savefig(metrics_file, dpi=100)
        plt.close()

        # mlflow artifact upload
        if self.mlflow_logger is not None:
            self.mlflow_logger.log_artifact(metrics_file, artifact_path="metrics")
            print(f"Bar chart saved & logged to MLflow: {metrics_file}")

    def save_anomaly_scores_as_csv(self,
                                   epoch: int,
                                   class_labels: torch.Tensor,
                                   anomaly_labels: torch.Tensor,
                                   p_s: torch.Tensor,
                                   z_s: torch.Tensor,
                                   p_t: torch.Tensor,
                                   z_t: torch.Tensor) -> str:
        save_dir = self.model_utils.get_save_dir()
        os.makedirs(f"{save_dir}/metric", exist_ok=True)
        csv_path = f"{save_dir}/metric/anomaly_scores_epoch{epoch}.csv"

        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["file_name", "class_label", "anomaly_label", "anomaly_score"])

            num_samples = p_s.shape[0]
            for i in range(num_samples):
                score = self.anomaly_score.calc_simsiam_anomaly_score(
                    p_s[i], z_t[i],
                    p_t[i], z_s[i]
                )
                c_label = class_labels[i].item()
                a_label = anomaly_labels[i].item()

                writer.writerow([c_label, a_label, float(score)])

        # mlflow artifact upload
        if self.mlflow_logger is not None:
            self.mlflow_logger.log_artifact(csv_path, artifact_path="metrics")
            print(f"Bar chart saved & logged to MLflow: {csv_path}")