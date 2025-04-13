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
        match_strategy (str): Strategy for matching source and target data. Options: 'sequential', 'random', 'pairwise'
    """
    def __init__(self, source_dataset, target_dataset, match_strategy='sequential'):
        super().__init__()
        self.source_dataset = source_dataset
        self.target_dataset = target_dataset
        self.match_strategy = match_strategy

        self.source_length = len(self.source_dataset)
        self.target_length = len(self.target_dataset)

    def __len__(self):
        return max(self.source_length, self.target_length)

    def __getitem__(self, idx):
        if self.match_strategy == 'random':
            src_idx = random.randint(0, self.source_length - 1)
            tgt_idx = random.randint(0, self.target_length - 1)
        elif self.match_strategy in ('sequential', 'pairwise'):
            src_idx = idx % self.source_length
            tgt_idx = idx % self.target_length
        else:
            raise ValueError(f"Unknown match strategy: {self.match_strategy}")

        src_data, src_label = self.source_dataset[src_idx]
        tgt_data, tgt_label = self.target_dataset[tgt_idx]

        return (src_data, src_label), (tgt_data, tgt_label)