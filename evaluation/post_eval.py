# evaluation/post_eval.py

import torch
import torch.nn as nn
import torch.nn.functional as F

from evaluation.modules.distribution_distance import get_boundary
from evaluation.modules.anomaly_metrics import calc_accuracy, calc_precision, calc_recall, calc_f1, calc_auc

class AnomalyScore(nn.Module):
    """
    Class for calculating anomaly detection scores using two approaches:
    1) SimSiam loss-based:
        For identical/normal s-t pairs, cosine similarity is high, resulting in low loss
        In the presence of anomalies, cosine similarity is low, leading to higher loss
        To compute per-sample scores, modify the core logic of SimSiamLoss (neg_cosine_similarity)
          to return values for individual samples instead of averaging over the batch
    2) Embedding distribution-based threshold:
        Set a percentile (e.g., 99%) threshold from the distribution of normal source + normal target embeddings
        Distances exceeding this threshold are classified as anomalies
    """
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.percentile = cfg["distribution"]["percentile"]

        self.fitted = False
        self.dist_threshold = None

    def calc_simsiam_anomaly_score(self, p1: torch.Tensor, z2: torch.Tensor, p2: torch.Tensor, z1: torch.Tensor) -> torch.Tensor:
        """
        Calc SimSiam anomaly score per-sample
        """
        p1_norm = F.normalize(p1, dim=1)  # (B, feat_dim)
        z2_norm = F.normalize(z2.detach(), dim=1)
        p2_norm = F.normalize(p2, dim=1)
        z1_norm = F.normalize(z1.detach(), dim=1)

        # per-sample neg cosine similarity
        loss_12 = -(p1_norm * z2_norm).sum(dim=1)  # (B,)
        loss_21 = -(p2_norm * z1_norm).sum(dim=1)

        simsiam_scores = 0.5 * (loss_12 + loss_21)  # (B,)

        return simsiam_scores
    
    def fit_distribution(self, source_normal_feats: torch.Tensor, target_normal_feats: torch.Tensor):
        """
        Fit a distribution (or set a threshold) based on normal source/target embeddings.
        Args
        source_normal_feats (torch.Tensor): Embeddings of normal source data, e.g., shape (N_s, feature_dim)
        target_normal_feats (torch.Tensor): Embeddings of normal target data, e.g., shape (N_t, feature_dim)
        """
        embeddings = torch.cat([source_normal_feats, target_normal_feats], dim=0)
        self.center = embeddings.mean(dim=0, keepdim=True)  # (1, feature_dim)

        # calc boundary
        self.dist_threshold = get_boundary(embeddings=embeddings, center=self.center, percentile=self.percentile)
        self.fitted = True

    def calc_distribution_anomaly_score(self, feat: torch.Tensor) -> torch.Tensor:
        """
        Calculate anomaly scores using a distribution-based threshold.
        Args
            feat (torch.Tensor): Embeddings extracted from the model, e.g., shape (batch_size, feature_dim)
        Returns
            anomaly_score (torch.Tensor): Per-sample anomaly scores (batch_size,). E.g., scores are higher for distances exceeding the threshold
        """
        if not self.fitted or self.dist_threshold is None:
            raise RuntimeError("[AnomalyDetector] Call fit_distribution() first.")
        
        # distance = feat - self.center
        distance = torch.norm(feat - self.center, p=2, dim=1)  # (batch_size, )
        anomaly_score = nn.functional.relu(distance - self.dist_threshold)

        return anomaly_score

def find_best_threshold(scores: torch.Tensor, labels: torch.Tensor, num_steps: int = 100):
    """
    Search for the threshold that maximizes the AUC metric, given 'scores' and 'labels'.
    Args:
        scores: Shape (B,)
        labels: Shape (B,) (0 or 1)
        num_steps: Number of threshold candidates (consider performance for large datasets)
    Returns:
        best_th (float): threshold that yields the best AUC
        best_auc (float): AUC value at that threshold
    """
    # convert scores to CPU tensor
    s_cpu = scores.detach().cpu()
    l_cpu = labels.detach().cpu()

    # generate threshold candidates at uniform intervals within the min-max range
    s_min = torch.min(s_cpu).item()
    s_max = torch.max(s_cpu).item()
    if s_min == s_max:
        # if all scores are identical, the threshold is meaningless
        preds = (s_cpu > s_min).long()
        auc_val = calc_auc(preds, l_cpu)
        return s_min, float(auc_val)

    thresholds = torch.linspace(s_min, s_max, steps=num_steps)
    best_th = 0.0
    best_auc = 0.0

    for th in thresholds:
        preds = (s_cpu > th).long()
        auc_val = calc_auc(preds, l_cpu)
        if auc_val > best_auc:
            best_auc = auc_val
            best_th = th.item()
    
    return best_th, best_auc

class AnomalyDetector:
    """
    Obtain anomaly scores from AnomalyScore (SimSiam/Distribution-based),
    automatically find the best threshold using the data to determine (normal/anomaly) predicted labels,
    and calculate Accuracy, Precision, Recall, F1, AUC using functions from anomaly_metrics.py.
    """
    def __init__(self, cfg):
        """
        cfg: Configuration dictionary
        """
        self.cfg = cfg
        self.anomaly_score_fn = AnomalyScore(cfg)

    def fit_distribution(self, source_feats_normal, target_feats_normal):
        """
        Set distribution-based information (center, dist_threshold, etc.) using normal s/t embeddings
        """
        self.anomaly_score_fn.fit_distribution(source_feats_normal, target_feats_normal)

    def evaluate_simsiam(self, p1: torch.Tensor, z2: torch.Tensor, p2: torch.Tensor, z1: torch.Tensor, anomaly_labels: torch.Tensor):
        """
        SimSiam-based anomaly score -> auto threshold -> (preds vs labels) -> metrics
        """
        # anomaly score
        scores = self.anomaly_score_fn.calc_simsiam_anomaly_score(p1, z2, p2, z1)  # (B,)

        # find best threshold (F1-based)
        best_th, best_auc = find_best_threshold(scores, anomaly_labels, num_steps=100)

        # Calculate predicted labels (0 or 1) using the threshold
        preds = (scores > best_th).long()  # (B,)

        # metric
        acc   = calc_accuracy(preds, anomaly_labels)
        prec  = calc_precision(preds, anomaly_labels)
        rec   = calc_recall(preds, anomaly_labels)
        f1    = calc_f1(preds, anomaly_labels)
        auc_v = calc_auc(scores, anomaly_labels)

        # dict
        metrics_dict = {
            "BestThreshold": best_th,
            "AUC_at_BestTh": best_auc,
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1": f1,
            "AUC": auc_v
        }
        return metrics_dict

    def evaluate_distribution(self, feats: torch.Tensor, labels: torch.Tensor):
        """
        Distribution-based anomaly score -> auto threshold -> (preds vs labels) -> metrics
        """
        # anomaly score
        distance = self.anomaly_score_fn.calc_distribution_anomaly_score(feats)  # (B,)

        # find best threshold (F1-based)
        best_th, best_auc = find_best_threshold(distance, labels, num_steps=100)

        preds = (distance > best_th).long()
        
        # metric
        acc   = calc_accuracy(preds, labels)
        prec  = calc_precision(preds, labels)
        rec   = calc_recall(preds, labels)
        f1    = calc_f1(preds, labels)
        auc_v = calc_auc(distance, labels)  # AUC using distance (or scores)

        metrics_dict = {
            "BestThreshold": best_th,
            "AUC_at_BestTh": best_auc,
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1": f1,
            "AUC": auc_v
        }
        return metrics_dict

