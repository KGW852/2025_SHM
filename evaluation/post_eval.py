# evaluation/post_eval.py

import torch
import torch.nn as nn
import torch.nn.functional as F

from evaluation.modules.distribution import get_boundary
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
    
class AnomalyDetector:
    """
    - AnomalyScore (SimSiam/Distribution 기반)로부터 anomaly score를 얻어온 뒤
    - 실제 라벨과 비교하여 (Threshold > score ? 정상/이상) 예측 라벨을 결정
    - anomaly_metrics.py의 함수들을 사용해 Accuracy, Precision, Recall, F1, AUC 등을 계산
    """
    def __init__(self, cfg):
        """
        Args
        ----
        cfg : dict
            설정/하이퍼파라미터 등을 담은 설정
            예) cfg["evaluation"]["threshold_simsiam"] = 0.3
                cfg["evaluation"]["threshold_distribution"] = 0.1
                ...
        """
        self.cfg = cfg
        self.anomaly_score_fn = AnomalyScore(cfg)  # 위에서 정의한 클래스 사용
        self.simsiam_threshold = cfg["evaluation"].get("threshold_simsiam", 0.5)
        self.dist_threshold   = cfg["evaluation"].get("threshold_distribution", 0.0)
        # dist_threshold는 fit_distribution으로 계산되므로, 여기서는 기본값만 가져오도록 하거나
        # 또는 percentile 기반이므로 별도 설정은 안 해도 됩니다.

    def fit_distribution(self, source_feats_normal, target_feats_normal):
        """
        정상 s/t embedding을 사용해 분포 기반 threshold 세팅
        """
        self.anomaly_score_fn.fit_distribution(source_feats_normal, target_feats_normal)

    def evaluate_simsiam(
        self, 
        p1: torch.Tensor, z2: torch.Tensor, 
        p2: torch.Tensor, z1: torch.Tensor,
        labels: torch.Tensor
    ):
        """
        SimSiam 기반 anomaly score를 얻어 이진분류(정상/이상) 성능을 평가
        Args
        ----
        p1, z2, p2, z1 : (B, feat_dim)
            모델 내부에서 뽑은 projection/prediction head 결과
        labels : (B,)
            실제 라벨 (0=normal, 1=anomaly 등)
        """
        # 1) anomaly score 계산
        scores = self.anomaly_score_fn.calc_simsiam_anomaly_score(p1, z2, p2, z1)  # (B,)

        # 2) threshold를 이용해 예측 라벨(0 or 1) 계산
        #    예: anomaly_score > simsiam_threshold 이면 1(이상), 아니면 0(정상)
        preds = (scores > self.simsiam_threshold).long()  # (B,)

        # 3) 지표 계산
        acc   = calc_accuracy(preds, labels)
        prec  = calc_precision(preds, labels)
        rec   = calc_recall(preds, labels)
        f1    = calc_f1(preds, labels)
        auc_v = calc_auc(scores, labels)  # AUC는 점수 기반이므로 scores, labels 필요

        # 결과를 딕셔너리 형태로 묶어서 반환
        metrics_dict = {
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1": f1,
            "AUC": auc_v
        }
        return metrics_dict

    def evaluate_distribution(self, feats: torch.Tensor, labels: torch.Tensor):
        """
        분포 기반 anomaly score를 얻어 이진분류(정상/이상) 성능을 평가
        Args
        ----
        feats : (B, feat_dim)
            모델에서 추출된 임베딩
        labels: (B,)
            실제 라벨(0=normal, 1=anomaly 등)
        """
        # 1) anomaly score 계산
        scores = self.anomaly_score_fn.calc_distribution_anomaly_score(feats)  # (B,)

        # 2) thresholding
        #    distribution 기반 anomaly_score는 이미 (distance - dist_threshold)의 ReLU 형태이므로,
        #    score가 0보다 크다는 것은 dist_threshold보다 distance가 크다는 뜻(즉 anomaly).
        #    -> (score > 0) 이면 anomaly
        preds = (scores > 0).long()

        # 3) 지표 계산
        acc   = calc_accuracy(preds, labels)
        prec  = calc_precision(preds, labels)
        rec   = calc_recall(preds, labels)
        f1    = calc_f1(preds, labels)
        auc_v = calc_auc(scores, labels)

        metrics_dict = {
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1": f1,
            "AUC": auc_v
        }
        return metrics_dict