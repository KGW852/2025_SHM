# evaluation/modules/all_scores.py

import torch
import torch.nn.functional as F

import numpy as np

from utils.csv_utils import save_csv


class ExtractScore:
    """
    All scores with labels per-sample extractor
    """
    def __init__(self, cfg):
        self.cfg = cfg
        self.recon_type = cfg["anomaly"]["recon_type"]

    def recon_scores(self, csv_path, results_list):
        file_name = [res["file_name"] for res in results_list]
        class_labels = [res["class_label"] for res in results_list]
        anomaly_labels = [res["anomaly_label"] for res in results_list]
        x_tensor = torch.tensor(np.array([res["x"] for res in results_list]))
        x_recon_tensor = torch.tensor(np.array([res["x_recon"] for res in results_list]))

        # calc score
        if self.recon_type == 'mse':
            p_scores = F.mse_loss(x_recon_tensor, x_tensor, reduction='none')
        else:  # self.loss_type == 'mae'
            p_scores = F.l1_loss(x_recon_tensor, x_tensor, reduction='none')
        scores = p_scores.view(p_scores.size(0), -1).sum(dim=1)  # (N,)

        # save score to csv per data
        rows_data = [["filename", "class_label", "anomaly_label", "score"]]
        for filename, class_label, anomaly_label, score in zip(file_name, class_labels, anomaly_labels, scores):
            rows_data.append([filename, int(class_label), int(anomaly_label), float(score)])

        save_csv(rows_data, csv_path)

    def distance_scores(self, csv_path, results_list):
        file_name = [res["file_name"] for res in results_list]
        class_labels = [res["class_label"] for res in results_list]
        anomaly_labels = [res["anomaly_label"] for res in results_list]
        distances = [res["distance"] for res in results_list]

        # save score to csv per data
        rows_data = [["filename", "class_label", "anomaly_label", "score"]]
        for filename, class_label, anomaly_label, score in zip(file_name, class_labels, anomaly_labels, distances):
            rows_data.append([filename, int(class_label), int(anomaly_label), float(score)])

        save_csv(rows_data, csv_path)
        
