# utils/model_utils.py

import os
import re


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
            self.model_name = cfg['model']['model_name']
            self.version = cfg['model']['version']
            self.suffix = cfg['model']['suffix']
            self.model_base_dir = cfg['model']['base_dir']
        else:  # dataclass
            self.model_name = cfg.model.name
            self.version = cfg.model.version
            self.suffix = cfg.model.suffix
            self.model_base_dir = cfg.model.base_dir

    def get_model_name(self) -> str:
        model_name = f"{self.model_name}_v{self.version}"
        return model_name
    
    def get_file_name(self, epoch: int) -> str:
        file_name = f"checkpoint_epoch{epoch}"
        file_name += self.suffix
        return file_name
    
    def get_save_dir(self):
        save_dir = f"{self.model_base_dir}/{self.model_name}/v{self.version}"
        os.makedirs(save_dir, exist_ok=True)
        return save_dir
    
    def get_file_path(self, file_name: str)-> str:
        save_dir = f"{self.model_base_dir}/{self.model_name}/v{self.version}"
        os.makedirs(save_dir, exist_ok=True)

        save_path = os.path.join(save_dir, file_name)
        return save_path

    def _get_available_epochs(self) -> list:
        """
        Extract and return the X value (integer) from checkpoint_epochX.pth files in the model save directory.
        """
        dir_path = self.get_save_dir()
        all_files = os.listdir(dir_path)
        pattern = rf"checkpoint_epoch(\d+){re.escape(self.suffix)}$"
        
        epochs = []
        for filename in all_files:
            match = re.match(pattern, filename)
            if match:
                ep = int(match.group(1))
                epochs.append(ep)
        
        epochs.sort()
        return epochs
    
    def get_run_epochs(self, run_epoch) -> list:
        """
        Generate and return a list of epochs for evaluation/testing.
        Args:
            [-1] : Use all epochs present in the save directory
            [start:end]: Slicing; range(start, end)
            [int, int, int]: List of specified epochs
            int: Single epoch
        Returns:
            list of int: Epoch numbers to be used for final evaluation/testing
        """
        available_epochs = self._get_available_epochs()
        
        # [-1]: all epoch
        if run_epoch == [-1]:
            return available_epochs
        
        # [start:end]
        if isinstance(run_epoch, str):
            m = re.match(r"\[(\d+):(\d+)\]", run_epoch.strip())
            if m:
                start_val = int(m.group(1))
                end_val = int(m.group(2))
                desired_list = list(range(start_val, end_val))
                final_epochs = [ep for ep in desired_list if ep in available_epochs]  # filtering
                return final_epochs
            else:
                return []

        # list
        if isinstance(run_epoch, list):
            final_epochs = [ep for ep in run_epoch if ep in available_epochs]
            return final_epochs

        # int
        if isinstance(run_epoch, int):
            if run_epoch in available_epochs:
                return [run_epoch]
            else:
                return []

        return []