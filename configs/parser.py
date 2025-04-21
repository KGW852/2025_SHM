# configs/parser.py

import os
import yaml
from copy import deepcopy

def read_config_file(path_or_dict):
    """
    Read a YAML configuration file or dictionary.
    """
    if isinstance(path_or_dict, dict):
        return path_or_dict
    elif isinstance(path_or_dict, str):
        with open(path_or_dict, 'r') as f:
            return yaml.safe_load(f)
    else:
        raise ValueError(f"Invalid config input: {path_or_dict}")

def merge_dicts(base_dict, extra_dict):
    """
    Overwrite extra_dict content on base_dict (recursive).
    """
    merged = deepcopy(base_dict)
    for k, v in extra_dict.items():
        if (
            k in merged 
            and isinstance(merged[k], dict) 
            and isinstance(v, dict)
        ):
            merged[k] = merge_dicts(merged[k], v)
        else:
            merged[k] = v
    return merged

def parse_config(default_config_path='configs/default.yaml'):
    """
    1) default.yaml load
    2) If the data_name key exists in default.yaml, 
       find and merge the .yaml file with the same name (configs/{data_name}.yaml)
    3) Return the final dictionary
    """
    # load default.yaml
    default_cfg = read_config_file(default_config_path)

    # If data_name is set, additionally merge the dataset YAML
    data_name = default_cfg.get('model', {}).get('data_name', None)
    if data_name:
        dataset_cfg_path = os.path.join('configs', f'{data_name}.yaml')
        if os.path.exists(dataset_cfg_path):
            dataset_cfg = read_config_file(dataset_cfg_path)
            final_cfg = merge_dicts(default_cfg, dataset_cfg)
            return final_cfg
        
    return default_cfg

# test code
"""
if __name__ == "__main__":
    config = parse_config()
    print("Final Config:\n", config)
"""