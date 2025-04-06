# dataloaders/data_loader.py

import os
import torch
from torch.utils.data import DataLoader

from .data_set import CustomDataset, DomainDataset

def get_train_loader(args):
    """
    Get the training data loader.
    Args:
        args (argparse.Namespace): command line arguments
    """
    # source
    train_src_dir = args.train_src_dir
    source_file_list = os.listdir(train_src_dir)
    source_file_list.sort()
    source_file_list = [os.path.join(train_src_dir, f) for f in source_file_list]

    # target
    train_tgt_dir = args.train_tgt_dir
    target_file_list = os.listdir(train_tgt_dir)
    target_file_list.sort()
    target_file_list = [os.path.join(train_tgt_dir, f) for f in target_file_list]

    # Create datasets
    source_dataset = CustomDataset(
        data_name=args.data_name,
        file_paths=source_file_list,
        transform=None
    )
    target_dataset = CustomDataset(
        data_name=args.data_name,
        file_paths=target_file_list,
        transform=None
    )
    da_dataset = DomainDataset(
        source_dataset,
        target_dataset,
        match_strategy=args.match_strategy
    )
    train_loader = DataLoader(
        da_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.n_workers,
        drop_last=True
    )

    return train_loader


def get_eval_loader(args):
    """
    Get the evaluation data loader.
    Args:
        args (argparse.Namespace): command line arguments
    """
    # Source (Evaluation) directory
    eval_src_dir = args.eval_src_dir
    source_file_list = os.listdir(eval_src_dir)
    source_file_list.sort()
    source_file_list = [os.path.join(eval_src_dir, f) for f in source_file_list]

    # Target (Evaluation) directory
    eval_tgt_dir = args.eval_tgt_dir
    target_file_list = os.listdir(eval_tgt_dir)
    target_file_list.sort()
    target_file_list = [os.path.join(eval_tgt_dir, f) for f in target_file_list]

    # Create datasets
    source_dataset = CustomDataset(
        data_name=args.data_name,
        file_paths=source_file_list,
        transform=None
    )
    target_dataset = CustomDataset(
        data_name=args.data_name,
        file_paths=target_file_list,
        transform=None
    )
    da_dataset = DomainDataset(
        source_dataset,
        target_dataset,
        match_strategy=args.match_strategy  # "random", "sequential", etc.
    )
    eval_loader = DataLoader(
        da_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.n_workers,
        drop_last=False
    )

    return eval_loader

def get_test_loader(args):
    """
    Get the test data loader.
    Args:
        args (argparse.Namespace): command line arguments
    """
    # Source (Test) directory
    test_src_dir = args.test_src_dir
    source_file_list = os.listdir(test_src_dir)
    source_file_list.sort()
    source_file_list = [os.path.join(test_src_dir, f) for f in source_file_list]

    # Target (Test) directory
    test_tgt_dir = args.test_tgt_dir
    target_file_list = os.listdir(test_tgt_dir)
    target_file_list.sort()
    target_file_list = [os.path.join(test_tgt_dir, f) for f in target_file_list]

    # Create datasets
    source_dataset = CustomDataset(
        data_name=args.data_name,
        file_paths=source_file_list,
        transform=None
    )
    target_dataset = CustomDataset(
        data_name=args.data_name,
        file_paths=target_file_list,
        transform=None
    )
    da_dataset = DomainDataset(
        source_dataset,
        target_dataset,
        match_strategy=args.match_strategy
    )
    test_loader = DataLoader(
        da_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.n_workers,
        drop_last=False
    )

    return test_loader