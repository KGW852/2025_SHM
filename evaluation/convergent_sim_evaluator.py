# evaluation/convergent_sim_evaluator.py

import os
import torch
import csv
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

from models.convergent_sim import ConvergentSim
from utils.model_utils import ModelUtils
from utils.logger import MLFlowLogger
from evaluation.modules.anomaly_metrics import AnomalyScore, AnomalyDetector
from evaluation.modules.umap import plot_latent_alignment

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
    def __init__(self, cfg: dict, mlflow_logger: MLFlowLogger, run_id: str, last_epoch: int, device: torch.device = None, final_center=None, final_radius=None):
        self.cfg = cfg
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.final_center = final_center
        self.final_radius = final_radius

        # model
        self.encoder = ConvergentSim(
            enc_in_dim=self.cfg["mlp"]["in_dim"],
            enc_hidden_dims=self.cfg["mlp"]["in_hidden_dims"],
            enc_latent_dim=self.cfg["mlp"]["latent_dim"],
            dropout=self.cfg["mlp"]["dropout"],
            proj_hidden_dim=self.cfg["sim"]["proj_hidden_dim"],
            proj_out_dim=self.cfg["sim"]["proj_out_dim"],
            pred_hidden_dim=self.cfg["sim"]["pred_hidden_dim"],
            pred_out_dim=self.cfg["sim"]["pred_out_dim"],
            svdd_latent_dim=self.cfg["svdd"]["latent_dim"]
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
        
        checkpoint = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        self.encoder.load_state_dict(checkpoint['encoder_state_dict'])
        self.encoder.eval()

        es_list, et_list = [], []
        zs_list, ps_list, zt_list, pt_list = [], [], [], []
        svdds_list, svddt_list = [], []
        cls_ys_list, cls_yt_list, ano_ys_list = [], [], []

        pbar = tqdm(enumerate(data_loader), total=len(data_loader), desc=f"[Test]  [Epoch {epoch}/{self.last_epoch}] | Metric: {self.method}", leave=False)
        with torch.no_grad():
            for batch_idx, data in pbar:
                (x_s, y_s), (x_t, y_t) = data  # y_s, y_t: tensor([class_label, anomaly_label])
                cls_ys = y_s[:,0]  # (B,)
                cls_yt = y_t[:,0]
                ano_ys = y_s[:,1]
                ano_yt = y_t[:,1]

                x_s = x_s.to(self.device)
                x_t = x_t.to(self.device)

                # forward
                es, et, zs, ps, zt, pt, svdds, svddt = self.encoder(x_s, x_t)

                # stack
                es_list.append(es.cpu())
                et_list.append(et.cpu())
                zs_list.append(zs.cpu())
                ps_list.append(ps.cpu())
                zt_list.append(zt.cpu())
                pt_list.append(pt.cpu())
                svdds_list.append(svdds.cpu())
                svddt_list.append(svddt.cpu())

                cls_ys_list.append(cls_ys.cpu())
                cls_yt_list.append(cls_yt.cpu())
                ano_ys_list.append(ano_ys.cpu())
                
        # tensor concat
        es_list = torch.cat(es_list, dim=0)
        et_list = torch.cat(et_list, dim=0)
        zs_list = torch.cat(zs_list, dim=0)
        ps_list = torch.cat(ps_list, dim=0)
        zt_list = torch.cat(zt_list, dim=0)
        pt_list = torch.cat(pt_list, dim=0)
        svdds_list = torch.cat(svdds_list, dim=0)
        svddt_list = torch.cat(svddt_list, dim=0)

        cls_ys_list = torch.cat(cls_ys_list, dim=0).long()
        cls_yt_list = torch.cat(cls_yt_list, dim=0).long()
        ano_ys_list = torch.cat(ano_ys_list, dim=0).long()
        """
        # debug
        unique_class_labels = torch.unique(cls_ys_list)
        print(f"[Debug] cls_ys_list shape: {unique_class_labels.shape}")
        print(f"[Debug] unique class labels: {unique_class_labels}")
        unique_anomaly_labels = torch.unique(ano_ys_list)
        print(f"[Debug] ano_ys_list shape: {unique_anomaly_labels.shape}")
        print(f"[Debug] unique ano_ys_list labels: {unique_anomaly_labels}")
        """
        # Evaluation: AnomalyDetector
        anomaly_dict = {}
        y_pred_opt, anomaly_scores = None, None
        
        ret = self.anomaly_detector.evaluate(p1=ps_list, z2=zt_list, p2=pt_list, z1=zs_list, 
                                             y_true=ano_ys_list, return_thresholded_preds=self.return_thresholded_preds)
        anomaly_dict, y_pred_opt, anomaly_scores = ret

        # (Optional) csv save
        self.save_anomaly_scores_as_csv(epoch=epoch, anomaly_labels=ano_ys_list, anomaly_scores=anomaly_scores, y_pred_opt=y_pred_opt)
        print(f"[Test]  [Epoch {epoch}/{self.last_epoch}] | Metric: {self.method} | ", anomaly_dict)

        return anomaly_dict, es_list, et_list, zs_list, zt_list, svdds_list, svddt_list, cls_ys_list, cls_yt_list

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
        (ano_dict_init, es_init, et_init, zs_init, zt_init, svdds_init, svddt_init, cls_ys_init, cls_yt_init) = self.test_epoch(test_loader, epoch=0)
        (ano_dict_final, es_final, et_final, zs_final, zt_final, svdds_final, svddt_final, cls_ys_final, cls_yt_final) = self.test_epoch(test_loader, epoch=self.last_epoch)

        # mlflow metrics log
        if self.mlflow_logger:
            self.mlflow_logger.log_metrics({f"init_{k}": v for k, v in ano_dict_init.items() if isinstance(v, (float, int))})
            self.mlflow_logger.log_metrics({f"final_{k}": v for k, v in ano_dict_final.items() if isinstance(v, (float, int))})

        self._plot_and_log_bar_chart(ano_dict_init, ano_dict_final)

        plot_latent_alignment(cfg=self.cfg, mlflow_logger=self.mlflow_logger, 
                              src_embed=es_final, tgt_embed=et_final, src_lbl=cls_ys_final, tgt_lbl=cls_yt_final, 
                              src_center=None, src_radian=None, 
                              epoch=self.last_epoch, f_name="test_encoder")
        plot_latent_alignment(cfg=self.cfg, mlflow_logger=self.mlflow_logger, 
                              src_embed=zs_final, tgt_embed=zt_final, src_lbl=cls_ys_final, tgt_lbl=cls_yt_final, 
                              src_center=None, src_radian=None, 
                              epoch=self.last_epoch, f_name="test_projector")
        plot_latent_alignment(cfg=self.cfg, mlflow_logger=self.mlflow_logger, 
                              src_embed=svdds_final, tgt_embed=svddt_final, src_lbl=cls_ys_final, tgt_lbl=cls_yt_final, 
                              src_center=self.final_center, src_radian=self.final_radius, 
                              epoch=self.last_epoch, f_name="test_svdd")

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