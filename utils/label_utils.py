# utils/label_utils.py

import os

def get_esc50_label(file_path):
    file_name = os.path.basename(file_path)
    if 'airplane' in file_name:
        return 47
    elif 'seawave' in file_name:
        return 11
    elif 'helicopter' in file_name:
        return 40
    else:
        return -1
    
def get_dongjak_label(file_path):
    file_name = os.path.basename(file_path)
    if 'normal' in file_name:
        return 0
    elif 's1' in file_name:
        return 1
    elif 's2' in file_name:
        return 2
    elif 'test18' in file_name:
        return 18
    elif 'test23' in file_name:
        return 23
    else:
        return -1
    
def get_anoshift_label(file_path):
    pass

def get_esc50_pseudo_label(file_path: str):
    class_label = get_esc50_label(file_path)
    if class_label in [47, 40]:
        anomaly_label = 0
    elif class_label == 11:
        anomaly_label = 1
    else:
        anomaly_label = -1
    return class_label, anomaly_label

def get_dongjak_pseudo_label(file_path: str):
    class_label = get_dongjak_label(file_path)
    if class_label == 0 or class_label in [18, 23]:
        anomaly_label = 0
    elif class_label in [1, 2]:
        anomaly_label = 1
    else:
        anomaly_label = -1
    return class_label, anomaly_label

def get_anoshift_pseudo_label(file_path: str):
    pass