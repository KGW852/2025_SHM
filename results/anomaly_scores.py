# results/anomaly_scores.py

import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc


plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "legend.fontsize": 10,
    "figure.dpi": 300,
    "axes.linewidth": 0.6,
    "lines.linewidth": 1.8,
    "grid.alpha": 0.25,
})

def _read_loss_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "score" in df.columns:
        df["score"] = pd.to_numeric(df["score"].astype(str).str.replace("'", ""), errors="coerce")
    return df

def plot_roc_curves(recon_path: str, dist_path: str, save_path: str):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    recon_df = _read_loss_csv(recon_path)
    dist_df = _read_loss_csv(dist_path)

    plt.figure(figsize=(8, 7))

    scenarios = [("Recon", recon_df),
                 ("Distance", dist_df)]

    # Pairs of classes to compare (normal class, abnormal class)
    class_pairs = [(0, 2), (18, 2), (23, 2)]

    for name, df in scenarios:
        for normal_class, anomaly_class_val in class_pairs:
            filtered_df = df[df["class_label"].isin([normal_class, anomaly_class_val])].copy()
            # y_true: anomaly_label (0: normal, 2: anormal)
            y_true = filtered_df["anomaly_label"]
            # y_score: predict score
            y_score = filtered_df["score"]

            # calc ROC curve and AUC
            fpr, tpr, _ = roc_curve(y_true, y_score)
            roc_auc = auc(fpr, tpr)

            label = f"{name} ({normal_class} vs {anomaly_class_val}) - AUC = {roc_auc:.4f}"
            plt.plot(fpr, tpr, label=label)

    # plot
    plt.plot([0, 1], [0, 1], color="navy", lw=1.8, linestyle="--", label="Random (AUC = 0.50)")
    plt.xlim([-0.05, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic (ROC) Curve")
    plt.legend(loc="lower right")
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    print(f"ROC curve plot saved to '{save_path}'")
    plt.close()


if __name__ == "__main__":
    recon_file = "./results/scores/csv/v3.2.4/s2(all)_scores_epoch30_recon.csv"
    dist_file = "./results/scores/csv/v3.2.4/s2(all)_scores_epoch30_distance.csv"
    output_file = "./results/scores/figure/v3.2.4/s2(all)_auc_roc_curves_final.png"

    plot_roc_curves(recon_path=recon_file, dist_path=dist_file, save_path=output_file)
