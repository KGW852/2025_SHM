# results/result_anomaly_scores.py

import os
import sys
import itertools
from typing import Dict, Tuple, Optional

import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc


WIDTH_MM = 75
HEIGHT_MM = 75
WIDTH_INCH = WIDTH_MM / 25.4
HEIGHT_IN = HEIGHT_MM / 25.4

plt.rcParams.update({
    "figure.figsize": (WIDTH_INCH, HEIGHT_IN),
    "figure.dpi": 600,
    "savefig.dpi": 600,
    #"figure.autolayout": True,
    "figure.subplot.hspace": 0.00,
    #"figure.subplot.bottom": 0.00,

    "font.family": "Times New Roman",
    "font.size": 8,
    "axes.titlesize": 8,
    "axes.labelsize": 8,
    "legend.fontsize": 8,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,

    "axes.linewidth": 1.0,
    "lines.linewidth": 1.0,
    "lines.markersize": 2,
    "grid.alpha": 0.25,
})

CLASS_PAIRS = [(0, 2), (18, 2), (23, 2)]

LEGEND_MAP = {
    ("Recon",     0, 2): "Healthy vs. Pneumonia",
    ("Recon",    18, 2): "Metal Noise vs. Pneumonia",
    ("Recon",    23, 2): "Device-B vs. Pneumonia",
    ("Distance",  0, 2): "FEM-based",
    ("Distance", 18, 2): "2018-measured",
    ("Distance", 23, 2): "2024-measured",
}

def _read_anomaly_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "score" in df.columns:
        df["score"] = pd.to_numeric(df["score"].astype(str).str.replace("'", ""), errors="coerce")
    return df

def plot_roc_curves(recon_path: str, dist_path: str, save_path: str):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    scenarios = []
    scenario_styles = {"Recon": ":", "Distance": "-"}

    if recon_path and os.path.isfile(recon_path):  # recon
        scenarios.append(("Recon", _read_anomaly_csv(recon_path)))
    else:
        print(f"[WARN] Recon CSV not found → skip: {recon_path}", file=sys.stderr)
    
    if dist_path and os.path.isfile(dist_path):  # dist
        scenarios.append(("Distance", _read_anomaly_csv(dist_path)))
    else:
        print(f"[WARN] Distance CSV not found → skip: {dist_path}", file=sys.stderr)

    if not scenarios:
        raise FileNotFoundError("No valid CSV files found for plotting.")

    colors = ["#0060df", "#ff6d00", "#008272"]
    #markers   = itertools.cycle(["o", "s", "^"])

    for idx, (normal_cls, anom_cls) in enumerate(CLASS_PAIRS):
        color = colors[idx]

        for name, df in scenarios:
            ls = scenario_styles[name]
            sub = df[df["class_label"].isin([normal_cls, anom_cls])]
            if sub.empty:
                continue
            y_true, y_score = sub["anomaly_label"], sub["score"]

            # calc ROC curve and AUC
            fpr, tpr, _ = roc_curve(y_true, y_score)
            roc_auc = auc(fpr, tpr)

            legend_text = LEGEND_MAP.get((name, normal_cls, anom_cls), f"{name} ({normal_cls} vs {anom_cls})")
            legend = f"{legend_text} (AUC = {roc_auc:.4f})"
            plt.plot(fpr, tpr, label=legend, color=color, linestyle=ls)

    # plot
    plt.plot([0, 1], [0, 1], linestyle="--", color='red', label="Random (AUC = 0.50)")
    plt.xlim([-0.05, 1.05])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    #plt.title("Receiver Operating Characteristic (ROC) Curve")
    plt.legend(frameon=False, loc="lower right")
    plt.grid(True, linestyle=":")

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    print(f"ROC curve plot saved to '{save_path}'")
    plt.close()


if __name__ == "__main__":
    recon_file = None
    #recon_file = "./results/anomaly scores/csv/v3.2.4/s2(all)_scores_epoch30_recon.csv"
    dist_file = "./results/anomaly scores/csv/v3.2.4/s2(all)_scores_epoch30_distance.csv"
    output_file = "./results/anomaly scores/figure/v3.2.4/s2(all)_auc_roc_curves_final.png"

    plot_roc_curves(recon_path=recon_file, dist_path=dist_file, save_path=output_file)
