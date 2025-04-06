# utils/model_utils.py

import os

class ModelUtils():
    """
    Managing model save, load and name
    Args:
        config (dict or object): 
            model.name (str)
            model.version (str)
    """
    def __init__(self, cfg):
        self.model_name = None
        self.version = None
        self.suffix = None
        self.model_base_dir = None

        if isinstance(cfg, dict):  # dict
            self.model_name = cfg['model']['name']
            self.version = cfg['model']['version']
            self.suffix = cfg['model']['suffix']
            self.model_base_dir = cfg['model']['base_dir']
        else:  # dataclass
            self.model_name = cfg.model.name
            self.version = cfg.model.version
            self.suffix = cfg.model.suffix
            self.model_base_dir = cfg.model.base_dir

    def get_model_name(self, epoch: int) -> str:
        file_name = f"checkpoint_epoch{epoch}"
        file_name += self.suffix
        return file_name
    
    def get_model_path(self, file_name: str)-> str:
        save_dir = f"{self.model_base_dir}/{self.model_name}/v{self.version}"
        os.makedirs(save_dir, exist_ok=True)

        save_path = os.path.join(save_dir, file_name)
        return save_path
