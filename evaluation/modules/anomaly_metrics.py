# evaluation/modules/anomaly_metrics.py

import torch
import numpy as np
from sklearn.metrics import roc_auc_score

def calc_accuracy(preds: torch.Tensor, labels: torch.Tensor) -> float:
    """
    Accuracy = (TP + TN) / (TP + TN + FP + FN)
    """
    # preds, labels: 0 or 1 (B,) tensor
    correct = (preds == labels).sum().item()
    total = len(labels)
    return correct / total if total > 0 else 0.0

def calc_precision(preds: torch.Tensor, labels: torch.Tensor) -> float:
    """
    Precision = TP / (TP + FP)
    """
    tp = ((preds == 1) & (labels == 1)).sum().item()
    fp = ((preds == 1) & (labels == 0)).sum().item()
    if (tp + fp) == 0:
        return 0.0
    return tp / (tp + fp)

def calc_recall(preds: torch.Tensor, labels: torch.Tensor) -> float:
    """
    Recall = TP / (TP + FN)
    """
    tp = ((preds == 1) & (labels == 1)).sum().item()
    fn = ((preds == 0) & (labels == 1)).sum().item()
    if (tp + fn) == 0:
        return 0.0
    return tp / (tp + fn)

def calc_f1(preds: torch.Tensor, labels: torch.Tensor) -> float:
    """
    F1 score = 2 * (precision * recall) / (precision + recall)
    """
    prec = calc_precision(preds, labels)
    rec = calc_recall(preds, labels)
    if (prec + rec) == 0.0:
        return 0.0
    return 2 * prec * rec / (prec + rec)

def calc_auc(scores: torch.Tensor, labels: torch.Tensor) -> float:
    """
    ROC AUC score
    """
    scores_np = scores.detach().cpu().numpy() if isinstance(scores, torch.Tensor) else scores
    labels_np = labels.detach().cpu().numpy() if isinstance(labels, torch.Tensor) else labels

    unique_labels = np.unique(labels_np)
    if len(unique_labels) < 2:
        return 0.5

    return roc_auc_score(labels_np, scores_np)