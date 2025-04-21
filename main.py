# main.py

import argparse
import random
import numpy as np
import torch
import importlib

from configs.config import Config
from dataloaders.data_loader import get_train_loader, get_eval_loader, get_test_loader
#from trainers.convergent_sim_trainer import ConvergentSimTrainer
#from evaluation.convergent_sim_evaluator import ConvergentSimEvaluator
from utils.logger import MLFlowLogger


def set_random_seed(seed: int):
    """
    random seed
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def run_experiment(exp_name: str):
    """
    Call the provided Config.expN() function to execute training
    Args:
        exp_name (str): 'exp1', 'exp2' ...
    """
    print(f"[INFO] Start experiment: {exp_name}")

    # config
    config_instance = getattr(Config, exp_name)()  # ex. Config.exp1(), Config.exp2()
    cfg = config_instance.config_dict

    # seed
    seed = cfg.get("seed", 42)
    set_random_seed(seed)

    # MLflow logger
    use_mlflow = cfg.get("mlflow", {}).get("use", False)
    if use_mlflow:
        tracking_uri = cfg.get("mlflow", {}).get("tracking_uri", "file:./mlruns")
        cfg_exp_name = cfg.get("exp_name", None)
        if cfg_exp_name == "all":
            experiment_name = exp_name  # exp1, exp2 ...
        else:
            experiment_name = cfg_exp_name if cfg_exp_name else exp_name
        mlflow_logger = MLFlowLogger(tracking_uri=tracking_uri, experiment_name=experiment_name)
    else:
        mlflow_logger = None

    # device
    device_str = cfg.get("device", "cuda")
    device = torch.device(device_str if torch.cuda.is_available() else 'cpu')

    # dynamic trainer
    trainer_module_name = cfg["model"]["trainer"].replace(".py", "")  # e.g. 'convergent_sim_trainer'
    trainer_module = importlib.import_module(f"trainers.{trainer_module_name}")
    TrainerClass = getattr(trainer_module, cfg["model"]["trainer_fn"])  # e.g. ConvergentSimTrainer

    trainer = TrainerClass(cfg, mlflow_logger, device=device)
    train_loader = get_train_loader(cfg)
    eval_loader = get_eval_loader(cfg)
    
    run_id, last_saved_epoch, final_center, final_radius = trainer.run(train_loader=train_loader, eval_loader=eval_loader, log_params_dict=cfg)

    # dynamic evaluator
    evaluator_module_name = cfg["model"]["evaluator"].replace(".py", "")
    evaluator_module = importlib.import_module(f"evaluation.{evaluator_module_name}")
    EvaluatorClass = getattr(evaluator_module, cfg["model"]["evaluator_fn"])
    
    evaluator = EvaluatorClass(cfg, mlflow_logger, run_id=run_id, last_epoch=last_saved_epoch, device=device, final_center=final_center, final_radius=final_radius)
    test_loader = get_test_loader(cfg)

    evaluator.run(eval_loader=eval_loader, test_loader=test_loader)


def main():
    cfg = Config.exp1().config_dict
    exp_name_from_yaml = cfg.get("exp_name", "exp1")

    # Automatically find the list of expN methods inside Config
    # Filter only methods starting with 'exp' from dir(Config)
    all_exps = [fn for fn in dir(Config) if fn.startswith("exp") and callable(getattr(Config, fn))]
    all_exps.sort()

    if exp_name_from_yaml == "all":
        for exp_name in all_exps:
            run_experiment(exp_name)
    else:
        if exp_name_from_yaml not in all_exps:
            raise ValueError(f"Experiment method '{exp_name_from_yaml}' not found in Config.")
        run_experiment(exp_name_from_yaml)


if __name__ == "__main__":
    main()