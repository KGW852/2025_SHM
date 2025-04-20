# configs/config.py

from dataclasses import dataclass, field
from .parser import parse_config

@dataclass
class Config:
    """
    A dataclass to store the hyperparameter dictionary loaded from parser.py as-is.
    Create an expN() method for each experiment.
    Configure it to return only the necessary key-value pairs with modifications.
    """
    config_dict: dict = field(default_factory=lambda: parse_config())

    @classmethod
    def exp1(cls):
        cfg = parse_config()
        cfg["model"]["version"] = 1.1
        return cls(config_dict=cfg)

    @classmethod
    def exp2(cls):
        cfg = parse_config()
        cfg["model"]["version"] = 2.0
        cfg["epochs"] = 3
        cfg["learning_rate"] = 5e-5
        return cls(config_dict=cfg)