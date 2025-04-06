# main.py

import types
import argparse

import torch

from configs.config import Config
from dataloaders.data_loader import get_train_loader, get_eval_loader
from trainers.convergent_sim_trainer import ConvergentSimTrainer

def main():
    """
    main: Config load → dataloader → trainer
    """
    # config load (exp1, exp2.)
    config_obj = Config.exp1()  # exp1
    cfg = config_obj.config_dict  # dict shape

    # device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Dataloader
    args = types.SimpleNamespace()
    args.train_src_dir = cfg["train"]["src_dir"]
    args.train_tgt_dir = cfg["train"]["tgt_dir"]
    args.eval_src_dir = cfg["eval"]["src_dir"]
    args.eval_tgt_dir = cfg["eval"]["tgt_dir"]

    args.data_name = cfg["data_name"]

    args.batch_size = cfg["batch_size"]
    args.n_workers = cfg["n_workers"]
    args.match_strategy = cfg["match_strategy"]


    train_loader = get_train_loader(args)
    eval_loader = get_eval_loader(args)

    # trainer
    trainer = ConvergentSimTrainer(cfg, device=device)
    trainer.run(train_loader, eval_loader=eval_loader)


if __name__ == "__main__":
    main()