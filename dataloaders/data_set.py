# dataloaders/data_set.py
import os
import random
import torch
from torch.utils.data import Dataset

from utils.csv_utils import read_csv
from utils.label_utils import get_esc50_pseudo_label, get_dongjak_pseudo_label, get_anoshift_pseudo_label

COLUMN_MAP = {
    'dongjak': (1, 2),
    'esc50': (0, None),
    'anoshift': (2, 5),
}

LABEL_FUNC_MAP = {
    'dongjak': get_dongjak_pseudo_label,
    'esc50': get_esc50_pseudo_label,
    'anoshift': get_anoshift_pseudo_label,
}


class CustomDataset(Dataset):
    """
    Args:
        data_name (str): dataset name, e.g., 'esc50', 'dongjak', 'anoshift'
        file_paths (List[str]): CSV file directory list
        transform (callable, optional): transform function to be applied to the data tensor.
    """
    def __init__(self, data_name, file_paths, transform=None):
        super().__init__()
        self.data_name = data_name
        self.file_paths = file_paths
        self.transform = transform

        if data_name not in COLUMN_MAP or data_name not in LABEL_FUNC_MAP:
            raise ValueError(f"Unknown data_name: {data_name}")

        self.ch1_col, self.ch2_col = COLUMN_MAP[data_name]
        self.label_func = LABEL_FUNC_MAP[data_name]

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        csv_path = self.file_paths[idx]
        rows = read_csv(csv_path, skip_header=True)

        ch1_list = []
        ch2_list = []
        for row in rows:
            ch1_val = float(row[self.ch1_col]) if self.ch1_col is not None else 0.0

            if self.ch2_col is None:
                ch2_val = 0.0
            else:
                ch2_val = float(row[self.ch2_col])
            
            ch1_list.append(ch1_val)
            ch2_list.append(ch2_val)

        ch1_tensor = torch.tensor(ch1_list, dtype=torch.float)  # shape: (N,)
        ch2_tensor = torch.tensor(ch2_list, dtype=torch.float)
        data_tensor = torch.stack([ch1_tensor, ch2_tensor], dim=0)  # shape: (2, N)

        if self.transform:  # transform
            data_tensor = self.transform(data_tensor)

        class_label, anomaly_label = self.label_func(csv_path)
        label_tensor = torch.tensor([class_label, anomaly_label], dtype=torch.long)

        return data_tensor, label_tensor
    
class DomainDataset(Dataset):
    """
    Args:
        source_dataset (CustomDataset): Source domain dataset
        target_dataset (CustomDataset): Target domain dataset
        match_strategy (str): Strategy for matching source and target data. Options: 'sequential', 'random'
        n_samples (int): Number of target samples to use in pairwise (e.g. 10)
    """
    def __init__(self, source_dataset, target_dataset, match_strategy='sequential', n_samples=10):
        super().__init__()
        self.source_dataset = source_dataset
        self.target_dataset = target_dataset
        self.match_strategy = match_strategy
        self.n_samples = n_samples

        self.source_length = len(self.source_dataset)
        self.target_length = len(self.target_dataset)

        self.pairwise_data = self._build_pair_samples()

    def _build_pair_samples(self):
        """
        In pairwise, pre-construct (src, tgt) pairs using n_samples items (or all),
        each class label in the source, select n_samples to match with the same targets.
        """
        source_data_by_label = {}
        for idx in range(self.source_length):
            s_data, s_label = self.source_dataset[idx]
            class_label = s_label[0].item()  # s_label = [class_label, anomaly_label]
            if class_label not in source_data_by_label:
                source_data_by_label[class_label] = []
            source_data_by_label[class_label].append((s_data, s_label))

        # sampling n_samples items per label (or all)
        for cl_label in source_data_by_label:
            if self.n_samples != -1 and len(source_data_by_label[cl_label]) > self.n_samples:
                source_data_by_label[cl_label] = source_data_by_label[cl_label][: self.n_samples]

        # sampling n_samples items from the target (or all)
        all_target_data = []
        for idx in range(self.target_length):
            t_data, t_label = self.target_dataset[idx]
            all_target_data.append((t_data, t_label))
        if self.n_samples != -1 and len(all_target_data) > self.n_samples:
            all_target_data = all_target_data[: self.n_samples]

        if len(all_target_data) == 0:
            return []

        # calculate the maximum length (L_max) of the label group
        label_group_lengths = [len(src_list) for src_list in source_data_by_label.values()]
        L_max = max(label_group_lengths) if label_group_lengths else 0

        # L_max-sized list pattern_idxs
        len_tgt = len(all_target_data)
        pattern_idxs = []
        for i in range(L_max):
            if self.match_strategy == 'sequential':
                pattern_idxs.append(i % len_tgt)
            elif self.match_strategy == 'random':
                pattern_idxs.append(random.randint(0, len_tgt - 1))
            else:
                raise ValueError(f"Unknown match strategy: {self.match_strategy}")
            
        # For each source label, perform 1:1 matching between the list of samples for that label and target_samples
        pairwise_data = []
        for cl_label, src_list in source_data_by_label.items():
            group_len = len(src_list)
            for i in range(group_len):
                src_data, src_label = src_list[i]
                if i < L_max:
                    tgt_idx = pattern_idxs[i]
                else:
                    if self.match_strategy == 'sequential':
                        tgt_idx = i % len_tgt
                    else:  # 'random'
                        tgt_idx = random.randint(0, len_tgt - 1)

                tgt_data, tgt_label = all_target_data[tgt_idx]
                pairwise_data.append(((src_data, src_label), (tgt_data, tgt_label)))

        return pairwise_data

    def __len__(self):
        return len(self.pairwise_data)

    def __getitem__(self, idx):
        if idx >= len(self.pairwise_data):
            raise IndexError("Index out of range for the constructed pairwise_data.")
        (src_data, src_label), (tgt_data, tgt_label) = self.pairwise_data[idx]

        return (src_data, src_label), (tgt_data, tgt_label)