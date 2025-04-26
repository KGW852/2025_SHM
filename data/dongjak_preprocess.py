# data/dongjak_preprocess.py

import os
import random
import shutil

from configs.config import Config
from utils.csv_utils import read_csv

from math import ceil
import numpy as np
import pandas as pd
import soundfile as sf
import librosa
import librosa.display
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split


def get_configs():
    config = Config.exp1()
    cfg = config.config_dict
    return cfg

