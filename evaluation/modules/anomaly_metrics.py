# evaluation/modules/anomaly_metrics.py

import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve


class AnomalyScore(nn.Module):
    """
    Class for calculating anomaly detection scores
    """
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.method = cfg["anomaly"]["method"]
        self.percentile = cfg["anomaly"]["distribution_percentile"]

        self.fitted = False
        self.dist_threshold = None
        self.dist_mean = None
        self.dist_std = None

    def simsiam_anomaly_score(self, p1: torch.Tensor, z2: torch.Tensor, p2: torch.Tensor, z1: torch.Tensor) -> torch.Tensor:
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

        simsiam_scores = 0.5 * (loss_12 + loss_21)  # shape (B,)

        return simsiam_scores
    
    def distance_anomaly_score(self, p1: torch.Tensor, z2: torch.Tensor, p2: torch.Tensor, z1: torch.Tensor) -> torch.Tensor:
        pass

    def distribution_anomaly_score(self, p1: torch.Tensor, z2: torch.Tensor, p2: torch.Tensor, z1: torch.Tensor) -> torch.Tensor:
        pass

    def anomaly_score(self, p1: torch.Tensor, z2: torch.Tensor, p2: torch.Tensor, z1: torch.Tensor) -> torch.Tensor:
        if self.method == 'simsiam':
            anomaly_scores = self.simsiam_anomaly_score(p1=p1, z2=z2, p2=p2, z1=z1)
        elif self.method == 'distance':
            anomaly_scores = self.distance_anomaly_score(p1=p1, z2=z2, p2=p2, z1=z1)
        elif self.method == 'distribution':
            anomaly_scores = self.distribution_anomaly_score(p1=p1, z2=z2, p2=p2, z1=z1)
        else:
            raise ValueError(f"Unknown anomaly detection method: {self.method}")

        return anomaly_scores

class AnomalyDetector:
    """
    Evaluate anomaly detection performance by finding an optimal threshold on continuous anomaly scores and computing metrics.
    Args:
        anomaly_score (AnomalyScore): An instance of the AnomalyScore class.
    """
    def __init__(self, cfg, anomaly_score: AnomalyScore):
        self.anomaly_score = anomaly_score
        self.best_threshold = None

    def evaluate(self, p1: torch.Tensor, z2: torch.Tensor, p2: torch.Tensor, z1: torch.Tensor, y_true: torch.Tensor, return_thresholded_preds: bool = False):
        """
        Anomaly score -> auto threshold -> (pred_label vs true_label) -> metrics
        """
        # anomaly score
        anomaly_scores = self.anomaly_score.anomaly_score(p1, z2, p2, z1)  # (B,)
        
        # tensor to numpy
        anomaly_scores = anomaly_scores.detach().cpu().numpy()
        y_true = y_true.detach().cpu().numpy()

        # calc ROC-AUC, FPR/TPR, threshold
        auc_val = roc_auc_score(y_true, anomaly_scores)
        fpr, tpr, thresholds = roc_curve(y_true, anomaly_scores, pos_label=1)

        best_f1 = -1
        best_threshold = 0
        for thr in thresholds:
            y_pred = (anomaly_scores >= thr).astype(int)
            f1v = f1_score(y_true, y_pred)
            if f1v > best_f1:
                best_f1 = f1v
                best_threshold = thr
        self.best_threshold = best_threshold

        y_pred_opt = (anomaly_scores >= best_threshold).astype(int)

        # metric
        acc = accuracy_score(y_true, y_pred_opt)
        prec = precision_score(y_true, y_pred_opt, zero_division=0)
        rec = recall_score(y_true, y_pred_opt, zero_division=0)
        f1v = f1_score(y_true, y_pred_opt, zero_division=0)

        results_dict = {
            'accuracy': acc,
            'precision': prec,
            'recall': rec,
            'f1': f1v,
            'auc': auc_val,
            'threshold': best_threshold
        }

        if return_thresholded_preds:
            return results_dict, y_pred_opt
        else:
            return results_dict
