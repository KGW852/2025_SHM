# dataloaders/data_set.py
import os
import random
import torch

from PIL import Image
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
        data_path (List[str]): CSV file directory list
        transform (callable, optional): transform function to be applied to the data tensor.
    """
    def __init__(self, data_name, data_path, transform=None):
        super().__init__()
        self.data_name = data_name
        self.data_path = data_path
        self.transform = transform

        if data_name not in COLUMN_MAP or data_name not in LABEL_FUNC_MAP:
            raise ValueError(f"Unknown data_name: {data_name}")

        self.ch1_col, self.ch2_col = COLUMN_MAP[data_name]
        self.label_func = LABEL_FUNC_MAP[data_name]

    def __len__(self):
        return len(self.data_path)

    def __getitem__(self, idx):
        csv_path = self.data_path[idx]
        file_name = os.path.basename(csv_path)
        rows = read_csv(csv_path, skip_header=True)

        ch1_list = []
        ch2_list = []
        for row in rows:
            ch1_val = float(row[self.ch1_col]) if self.ch1_col is not None else 0.0
            ch2_val = float(row[self.ch2_col]) if self.ch2_col is not None else 0.0
            
            ch1_list.append(ch1_val)
            ch2_list.append(ch2_val)

        ch1_tensor = torch.tensor(ch1_list, dtype=torch.float)  # shape: (N,)
        ch2_tensor = torch.tensor(ch2_list, dtype=torch.float)
        data_tensor = torch.stack([ch1_tensor, ch2_tensor], dim=0)  # shape: (2, N)

        if self.transform:  # transform
            data_tensor = self.transform(data_tensor)

        class_label, anomaly_label = self.label_func(csv_path)
        label_tensor = torch.tensor([class_label, anomaly_label], dtype=torch.long)

        return data_tensor, label_tensor, file_name

class ImgCustomDataset(Dataset):
    def __init__(self, data_name, data_path, transform=None):
        super().__init__()
        self.data_name = data_name
        self.data_path = data_path
        self.transform = transform

        if data_name not in LABEL_FUNC_MAP:
            raise ValueError(f"Unknown data_name: {data_name}")

        self.label_func = LABEL_FUNC_MAP[data_name]

        self.labels = []
        for path in self.data_path:
            class_label, anomaly_label = self.label_func(path)
            self.labels.append(torch.tensor([class_label, anomaly_label], dtype=torch.long))

    def __len__(self):
        return len(self.data_path)

    def __getitem__(self, idx):
        img_path = self.data_path[idx]
        file_name = os.path.basename(img_path)
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        label_tensor = self.labels[idx]

        return image, label_tensor, file_name


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

        self.pair_indices = self._build_pair_indices()

    def _build_pair_indices(self):
        """
        Build a list of (src_idx, tgt_idx) pairs based on n_samples and match_strategy.
        """
        source_indices_by_label = {}
        for idx in range(self.source_length):
            class_label = self.source_dataset.labels[idx][0].item()
            if class_label not in source_indices_by_label:
                source_indices_by_label[class_label] = []
            source_indices_by_label[class_label].append(idx)

        # select n_samples indices per class in source (or all if n_samples=-1)
        for cl_label in source_indices_by_label:
            if self.n_samples != -1 and len(source_indices_by_label[cl_label]) > self.n_samples:
                source_indices_by_label[cl_label] = source_indices_by_label[cl_label][:self.n_samples]

        # select n_samples target indices (or all if n_samples=-1)
        if self.n_samples != -1 and self.target_length > self.n_samples:
            selected_tgt_indices = list(range(self.n_samples))
        else:
            selected_tgt_indices = list(range(self.target_length))

        len_tgt = len(selected_tgt_indices)
        if len_tgt == 0:
            return []

        # calculate the maximum length (L_max) of the label group
        label_group_lengths = [len(src_list) for src_list in source_indices_by_label.values()]
        L_max = max(label_group_lengths) if label_group_lengths else 0

        # create pattern_idxs for pairing based on match_strategy
        pattern_idxs = []
        for i in range(L_max):
            if self.match_strategy == 'sequential':
                pattern_idxs.append(i % len_tgt)
            elif self.match_strategy == 'random':
                pattern_idxs.append(random.randint(0, len_tgt - 1))
            else:
                raise ValueError(f"Unknown match strategy: {self.match_strategy}")
            
        # build list of (source_index, target_index) pairs
        pairwise_indices = []
        for cl_label, src_indices in source_indices_by_label.items():
            for i, src_idx in enumerate(src_indices):
                tgt_pattern_idx = pattern_idxs[i]
                tgt_idx = selected_tgt_indices[tgt_pattern_idx]
                pairwise_indices.append((src_idx, tgt_idx))

        return pairwise_indices

    def __len__(self):
        return len(self.pair_indices)

    def __getitem__(self, idx):
        """
        Return actual data from source/target dataset by the pair of indices.
        """
        if idx >= len(self.pair_indices):
            raise IndexError("Index out of range for pair_indices.")

        src_idx, tgt_idx = self.pair_indices[idx]
        src_data, src_label, src_path = self.source_dataset[src_idx]
        tgt_data, tgt_label, tgt_path = self.target_dataset[tgt_idx]

        return (src_data, src_label, src_path), (tgt_data, tgt_label, tgt_path)
    