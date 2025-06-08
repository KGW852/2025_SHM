# evaluation/modules/histogram.py

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


class HistPlot:
    """
    Histogram plot anomaly scores distributions with per-class.
    """
    def  __init__(self, anomaly_dict):
        self.threshold = anomaly_dict['threshold']

    def _process(self, results):
        results = np.asarray(results)
        return results
        
    def plot_hist(self, save_path, scores, class_labels):
        scores = self._process(scores)
        class_labels = self._process(class_labels)
        if scores.shape[0] != class_labels.shape[0]:
            raise ValueError("`scores` and `class_labels` must have identical length.")
        
        # plot per label
        unique_labels = np.unique(class_labels)
        cmap = plt.get_cmap("tab10", len(unique_labels))

        sns.set_style("whitegrid")
        plt.figure(figsize=(8, 6))

        for i, lbl in enumerate(unique_labels):
            color = cmap(i)
            idx = (class_labels == lbl)
            class_scores = scores[idx]

            # Histogram curve
            sns.histplot(class_scores, bins=50, stat="density", multiple="layer", fill=True, alpha=0.4, color=color, label=f"Class {lbl}")

        # threshold line
        plt.axvline(x=self.threshold, color='red', linestyle="--", linewidth=1.2, label=f"Threshold = {self.threshold:.3f}")

        plt.title("Anomaly Score Distribution by Class (Histogram)")
        plt.xlabel("Anomaly Score")
        plt.ylabel("Density")
        plt.xlim(0, 0.5)
        plt.legend()
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()
