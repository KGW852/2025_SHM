# evaluation/modules/anomaly_metrics.py

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
from scipy.stats import gaussian_kde

from utils.csv_utils import save_csv


class AnomalyScore(nn.Module):
    """
    Class for calculating anomaly detection scores
    """
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.method = cfg["anomaly"]["method"]
        self.percentile = cfg["anomaly"]["distribution_percentile"]
        self.recon_type = cfg["anomaly"]["recon_type"]

        self.fitted = False
        self.dist_threshold = None
        self.dist_mean = None
        self.dist_std = None

    def simsiam_anomaly_score(self, z1: torch.Tensor, p1: torch.Tensor, z2: torch.Tensor, p2: torch.Tensor) -> torch.Tensor:
        """
        Calc SimSiam anomaly score per-sample
        """
        p1_norm = F.normalize(p1, dim=1)  # shape (B, latent_dim)
        z2_norm = F.normalize(z2.detach(), dim=1)
        p2_norm = F.normalize(p2, dim=1)
        z1_norm = F.normalize(z1.detach(), dim=1)

        # per-sample neg cosine similarity
        loss_12 = -(p1_norm * z2_norm).sum(dim=1)  # shape (B,)
        loss_21 = -(p2_norm * z1_norm).sum(dim=1)

        simsiam_score = 0.5 * (loss_12 + loss_21)  # shape (B,)

        return simsiam_score
    
    def distance_anomaly_score(self, distance: torch.Tensor):
        """
        SVDD distance score per-sample
        """
        distance_score = distance
        return distance_score

    def distribution_anomaly_score(self, features: torch.Tensor, center: torch.Tensor):
        pass

    def reconloss_anomaly_score(self, x_recon, x):
        if self.recon_type == 'mse':
            loss = F.mse_loss(x_recon, x, reduction='sum')
        else:  # self.loss_type == 'mae'
            loss = F.l1_loss(x_recon, x, reduction='sum')
        return loss

    def anomaly_score(self, **kwargs) -> torch.Tensor:
        if self.method == 'simsiam':
            z1 = kwargs.get('z1', None)
            p1 = kwargs.get('p1', None)
            z2 = kwargs.get('z2', None)
            p2 = kwargs.get('p2', None)
            if z1 is None or p1 is None or z2 is None or p2 is None:
                raise ValueError("For 'simsiam' method, please provide z1, p1, z2, p2.")
            anomaly_score = self.simsiam_anomaly_score(z1, p1, z2, p2)

        elif self.method == 'distance':
            distance = kwargs.get('distance', None)
            if distance is None:
                raise ValueError("For 'distance' method, please provide 'distance'.")
            anomaly_score = self.distance_anomaly_score(distance)

        elif self.method == 'distribution':
            pass

        elif self.method == 'reconloss':
            x = kwargs.get('x', None)
            x_recon = kwargs.get('x_recon', None)
            if x is None or x_recon is None:
                raise ValueError("For 'reconloss' method, please provide x, x_recon.")
            anomaly_score = self.reconloss_anomaly_score(x_recon, x)

        else:
            raise ValueError(f"Unknown anomaly detection method: {self.method}")

        return anomaly_score

class AnomalyMetric:
    """
    Evaluate accuracy, precision, recall, f1: classify metrics
    """
    def __init__(self, cfg, file_name, y_true, y_score):
        self.cfg = cfg
        self.thresholded = cfg["anomaly"].get("thresholded", None)

        self.file_name_arr = np.array(file_name, dtype=object)
        self.y_true_arr = np.array(y_true)
        self.y_score_arr = np.array(y_score)

        self.best_threshold = None
        self.results_dict = None

    def calc_metric(self):
        # calc ROC-AUC, FPR/TPR, threshold
        auc_val = roc_auc_score(self.y_true_arr, self.y_score_arr)

        # threshold
        if self.thresholded is None:  # ROC curve: fpr, tpr, thresholds
            fpr, tpr, roc_thresholds = roc_curve(self.y_true_arr, self.y_score_arr, pos_label=1)
            j_scores = tpr - fpr
            best_idx = np.argmax(j_scores)
            best_threshold = roc_thresholds[best_idx]
        else:  # user defined threshold(float)
            best_threshold = self.thresholded
        
        y_pred = (self.y_score_arr > best_threshold).astype(int)

        # metric
        acc = accuracy_score(self.y_true_arr, y_pred)
        prec = precision_score(self.y_true_arr, y_pred, zero_division=0)
        rec = recall_score(self.y_true_arr, y_pred, zero_division=0)
        f1v = f1_score(self.y_true_arr, y_pred, zero_division=0)

        self.results_dict = {
            'accuracy': acc,
            'precision': prec,
            'recall': rec,
            'f1': f1v,
            'auc': auc_val,
            'threshold': best_threshold
        }
        return self.results_dict

    def save_anomaly_scores_as_csv(self, data_csv_path, metric_csv_path):
        if self.y_true_arr is None or self.y_score_arr is None or self.results_dict is None:
            raise ValueError("No data to save. Please run process_metric first.")
        
        # save score to csv per data
        rows_data = [["filename", "label", "score"]]
        for filename, label, score in zip(self.file_name_arr, self.y_true_arr, self.y_score_arr):
            rows_data.append([filename, label, score])

        save_csv(rows_data, data_csv_path)

        # save metric to csv
        rows_metric = [["metric", "value"]]
        for k, v in self.results_dict.items():
            rows_metric.append([k, v])

        save_csv(rows_metric, metric_csv_path)
        