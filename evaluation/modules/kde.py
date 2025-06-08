# evaluation/modules/kde.py

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


class KDEPlot:
    """
    KDE plot anomaly scores distributions with per-class.
    """
    def  __init__(self, anomaly_dict):
        self.threshold = anomaly_dict['threshold']

    def _process(self, results):
        results = np.asarray(results)
        return results
        
    def plot_kde(self, save_path, scores, class_labels):
        scores = self._process(scores)
        class_labels = self._process(class_labels)
        if scores.shape[0] != class_labels.shape[0]:
            raise ValueError("scores and class_labels must have identical length.")
        
        # plot per label
        unique_labels = np.unique(class_labels)
        cmap = plt.get_cmap("tab10", len(unique_labels))

        sns.set_style("whitegrid")
        plt.figure(figsize=(8, 6))

        for i, lbl in enumerate(unique_labels):
            color = cmap(i)
            idx = (class_labels == lbl)
            class_scores = scores[idx]

            # KDE curve
            sns.kdeplot(class_scores, bw_adjust=0.5, cut=0, clip=(0, None), fill=True, alpha=0.4, linewidth=1.8, color=color, label=f"Class {lbl}")

        # threshold line
        plt.axvline(x=self.threshold, color='red', linestyle="--", linewidth=1.2, label=f"Threshold = {self.threshold:.3f}")

        plt.title("Anomaly Score Distribution by Class (KDE)")
        plt.xlabel("Anomaly Score")
        plt.ylabel("Density")
        plt.xlim(0, 0.5)
        plt.legend()
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()