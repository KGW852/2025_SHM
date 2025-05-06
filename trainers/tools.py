# trainers/tools.py

import torch.optim as optim
from torch.optim.lr_scheduler import (StepLR, MultiStepLR, ExponentialLR, CosineAnnealingLR)


class Optimizer:
    """
    Create various Optimizer instances based on the cfg.optim configuration values
    """
    def __init__(self, cfg):
        self.cfg = cfg

    def get_optimizer(self, model_params):
        opt_cfg = self.cfg["optimizer"]
        opt_type = opt_cfg["type"].lower()

        lr = float(opt_cfg["learning_rate"])
        weight_decay = float(opt_cfg.get("weight_decay", 0.0))
        momentum = opt_cfg.get("momentum", 0.0)

        if opt_type == "adam":
            optimizer = optim.Adam(model_params, lr=lr, weight_decay=weight_decay)

        elif opt_type == "adamw":
            optimizer = optim.AdamW(model_params, lr=lr, weight_decay=weight_decay)

        elif opt_type == "sgd":
            optimizer = optim.SGD(model_params, lr=lr, momentum=momentum, weight_decay=weight_decay)

        elif opt_type == "rmsprop":
            optimizer = optim.RMSprop(model_params, lr=lr, momentum=momentum, weight_decay=weight_decay)

        else:
            raise ValueError(f"unsupported optimizer: {opt_cfg['type']}")
        
        return optimizer
    
class Scheduler:
    """
    Create various Scheduler instances based on the cfg.scheduler configuration values
    """
    def __init__(self, cfg):
        self.cfg = cfg

    def get_scheduler(self, optimizer):
        sch_cfg = self.cfg["scheduler"]
        sch_type = sch_cfg["type"].lower()

        if sch_type == "steplr":  # step size, gamma
            step_size = sch_cfg["step_size"]
            gamma = sch_cfg["gamma"]
            scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)
        
        elif sch_type == "multisteplr":  # ex. milestones = [30, 80], gamma = 0.1
            milestones = sch_cfg["milestones"]
            gamma = sch_cfg["gamma"]
            scheduler = MultiStepLR(optimizer, milestones=milestones, gamma=gamma)
        
        elif sch_type == "exponentiallr":
            gamma = sch_cfg["gamma"]
            scheduler = ExponentialLR(optimizer, gamma=gamma)
        
        elif sch_type == "cosineannealinglr":  # ex. T_max = 50, eta_min=0
            t_max = sch_cfg["t_max"]
            eta_min = sch_cfg.get("eta_min", 0.0)
            scheduler = CosineAnnealingLR(optimizer, T_max=t_max, eta_min=eta_min)
        
        else:
            raise ValueError(f"unsupported scheduler: {sch_cfg['type']}")
        
        return scheduler

class EarlyStopperMetric:
    """
    Early stops the training if validation loss (or other monitored metric) does not improve after a given patience.
    Args:
        patience (int): How many epochs to wait after last improvement.(Default: 5)
        mode (str): One of ["min", "max"].
            "min": Expect the monitored metric to get smaller (e.g. val_loss).
            "max": Expect the monitored metric to get larger (e.g. accuracy). (Default: "min")
        min_delta (float): Minimum change in the monitored metric 
                           to be considered as improvement. (Default: 0.0)
        baseline (float): Baseline value for the metric. If provided, training will continue 
                          until the metric surpasses this baseline. (Default: None)
    """
    def __init__(self, patience=5, mode="min", min_delta=0.0, baseline=None):
        self.patience = patience
        self.mode = mode.lower()
        self.min_delta = min_delta
        self.baseline = baseline
        
        self.num_bad_epochs = 0
        self.is_stop = False
        
        if self.mode not in ["min", "max"]:
            raise ValueError("mode should be one of ['min', 'max']")

        # best score tracking
        self.best_score = None

    def step(self, metric_value):
        """
        baseline exists and mode='max', best_score may be less than or equal to the baseline
        baseline exists and mode='min', best_score may be greater than or equal to the baseline
        no baseline exists, set it to the current metric
        """
        if self.best_score is None:  # initial step or baseline set
            if self.baseline is not None:
                if self.mode == "min":
                    self.best_score = min(self.baseline, metric_value)
                else:  # mode == "max"
                    self.best_score = max(self.baseline, metric_value)
            else:
                self.best_score = metric_value
            return self.is_stop

        # improvement assessment
        if self.mode == "min":
            improvement = metric_value < (self.best_score - self.min_delta)
        else:  # 'max'
            improvement = metric_value > (self.best_score + self.min_delta)

        # if improvement, update best; if not, increment bad_epoch count
        if improvement:
            self.best_score = metric_value
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1
        
        # stop if patience is exceeded
        if self.num_bad_epochs >= self.patience:
            self.is_stop = True

        return self.is_stop

class EarlyStopper:
    def __init__(self, cfg):
        self.cfg = cfg

    def get_early_stopper(self):
        early_cfg = self.cfg.get("early_stopper", None)

        # return None: if use=False or if no early_stopper session exists
        if early_cfg is None or not early_cfg.get("use", False):
            return None

        patience = early_cfg.get("patience", 5)
        mode = early_cfg.get("mode", "min")
        min_delta = early_cfg.get("min_delta", 0.0)
        baseline = early_cfg.get("baseline", None)

        return EarlyStopperMetric(patience=patience, mode=mode, min_delta=min_delta, baseline=baseline)

class SimWarmUp:
    def __init__(self, cfg):
        self.cfg = cfg

    def get_scale(self, epoch):
        warmup_cfg = self.cfg.get("sim", None)
        start = warmup_cfg.get("warmup_start", 10)
        end = warmup_cfg.get("warmup_end", 15)

        if epoch < start:
            return 0.0
        elif epoch > end:
            return 1.0
        else:
            return (epoch - start + 1) / (end - start + 1)

class SVDDWarmUp:
    def __init__(self, cfg):
        self.cfg = cfg

    def get_scale(self, epoch):
        warmup_cfg = self.cfg.get("svdd", None)
        start = warmup_cfg.get("warmup_start", 10)
        end = warmup_cfg.get("warmup_end", 15)

        if epoch < start:
            return 0.0
        elif epoch > end:
            return 1.0
        else:
            return (epoch - start + 1) / (end - start + 1)