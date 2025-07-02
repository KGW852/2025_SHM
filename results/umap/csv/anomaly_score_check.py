# results/umap/csv/anomaly_score_check.py

import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


df = pd.read_csv("./results/umap/csv/v3.0_resnet_ae/umap_s2(18)_encoder_epoch5.csv")

df = df.rename(columns={
    df.columns[0]: "x1",
    df.columns[1]: "x2",
    df.columns[2]: "class_label",
    df.columns[3]: "anomaly_label"
})

# 0-2. 숫자형 변환 (문자 섞여도 NaN 처리)
num_cols = ["x1", "x2"]
df[num_cols] = df[num_cols].apply(pd.to_numeric, errors="coerce")
df[["class_label", "anomaly_label"]] = df[["class_label", "anomaly_label"]].apply(pd.to_numeric, errors="coerce")

# 0-3. 필수 열에 NaN 이 남은 행은 제거
df = df.dropna(subset=num_cols + ["class_label", "anomaly_label"])

# 1. class label 0, 18인 정상 데이터 중심(center) 계산
normal_mask = df["class_label"].isin([0, 18])
center = df.loc[normal_mask, ["x1", "x2"]].mean().values  # shape = (2,)
print(f"Center (x̄, ȳ) = {center}")

# 2. 모든 데이터에 대해 center와의 유클리드 거리 계산
df["distance"] = np.linalg.norm(df[["x1", "x2"]].values - center, axis=1)

# -------------------------------------------------------
# 3. distance를 바탕으로 class label 0 vs 2 성능 지표 산출
#    * 기준(threshold)은 '정상(training) 거리의 95‑퍼센타일'
#    * 필요 시 퍼센타일 값이나 방법을 조정해도 무방
# -------------------------------------------------------
# 3‑1) 평가 대상 행만 추출 (label 0 과 2)
eval_mask = df["class_label"].isin([0, 2])
eval_df = df.loc[eval_mask].copy()

# 3‑2) 임계값(threshold) 설정
threshold = np.percentile(df.loc[normal_mask, "distance"], 95)
print(f"Distance threshold (95th pct of normals) = {threshold:.4f}")

# 3‑3) 예측 레이블 생성: 거리 > threshold → anomaly(1), else normal(0)
eval_df["pred_label"] = (eval_df["distance"] > threshold).astype(int)

# 3‑4) 성능 지표 계산
y_true = eval_df["anomaly_label"]
y_pred = eval_df["pred_label"]
y_score = eval_df["distance"]      # AUC 계산용 연속 점수

metrics = {
    "Accuracy" : accuracy_score(y_true, y_pred),
    "Precision": precision_score(y_true, y_pred, zero_division=0),
    "Recall"   : recall_score(y_true, y_pred, zero_division=0),
    "F1-score" : f1_score(y_true, y_pred, zero_division=0),
    "AUC"      : roc_auc_score(y_true, y_score)
}

print("\n=== Evaluation (class 0 vs 2) ===")
for k, v in metrics.items():
    print(f"{k:9s}: {v:.4f}")

"""
num_cols = ["x1", "x2"]
df[num_cols] = df[num_cols].apply(pd.to_numeric, errors="coerce")
df[["class_label", "anomaly_label"]] = df[["class_label", "anomaly_label"]].apply(pd.to_numeric, errors="coerce")
df = df.dropna(subset=num_cols + ["class_label", "anomaly_label"])

# 1. 정상(class 0, 18) 중심(center) 계산
normal_mask = df["class_label"].isin([0, 18])
center = df.loc[normal_mask, ["x1", "x2"]].mean().values  # shape = (2,)
print(f"Center (x̄, ȳ) = {center}")

# 모든 데이터에 대해 Euclidean distance 계산
df["distance"] = np.linalg.norm(df[["x1", "x2"]].values - center, axis=1)

# 2. class label 0 vs 2 평가용 서브셋 추출
eval_mask = df["class_label"].isin([0, 2])
eval_df = df.loc[eval_mask].copy()

y_true  = eval_df["anomaly_label"].values
y_score = eval_df["distance"].values

# 3. ROC 곡선 기반 최적 임계값 (Youden J = TPR - FPR) 계산
fpr, tpr, roc_thresholds = roc_curve(y_true, y_score, pos_label=1)
j_scores       = tpr - fpr
best_idx       = np.argmax(j_scores)
best_threshold = roc_thresholds[best_idx]

print(f"ROC-AUC                     : {roc_auc_score(y_true, y_score):.4f}")
print(f"Best threshold (Youden J)   : {best_threshold:.6f}")
print(f"TPR@best_thr, FPR@best_thr  : {tpr[best_idx]:.4f}, {fpr[best_idx]:.4f}")

# 4. 최적 임계값으로 예측 & 지표 산출
y_pred = (y_score > best_threshold).astype(int)

metrics = {
    "Accuracy" : accuracy_score(y_true, y_pred),
    "Precision": precision_score(y_true, y_pred, zero_division=0),
    "Recall"   : recall_score(y_true, y_pred, zero_division=0),
    "F1-score" : f1_score(y_true, y_pred, zero_division=0),
    "AUC"      : roc_auc_score(y_true, y_score),
    "Threshold": best_threshold
}

print("\n=== Evaluation (class 0 vs 2) ===")
for k, v in metrics.items():
    # Threshold는 그대로, 나머지는 소수점 4째 자리까지 표시
    if k == "Threshold":
        print(f"{k:9s}: {v}")
    else:
        print(f"{k:9s}: {v:.4f}")
"""