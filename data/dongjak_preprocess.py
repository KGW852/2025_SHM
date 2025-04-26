# data/dongjak_preprocess.py

import os
import random

from configs.config import Config
from utils.csv_utils import read_csv, read_second_column
from data.tools_preprocess import Segmenter, NoiseData, BaselineFitter

from math import ceil
from collections import defaultdict
from sklearn.model_selection import train_test_split
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt


def get_configs():
    config = Config.exp1()
    cfg = config.config_dict
    return cfg

def _collect_paths(d):
    paths = []
    for v in d.values():
        if isinstance(v, dict):
            paths.extend(_collect_paths(v))
        elif isinstance(v, str):
            paths.append(v)
    return paths
    
class Preprocessor:
    """dongjak Acc-csv file to STFT-png & train/eval/test set"""
    def __init__(self, cfg: dict):
        self.cfg = cfg

        # original path
        orig = cfg["orig_dir"]
        self.orig_normal = orig["normal_dir"]
        self.orig_s1 = orig["s1_dir"]
        self.orig_s2 = orig["s2_dir"]
        self.orig_test = orig["test_dir"]

        # save path
        self.train_dirs = cfg["train_dir"]
        self.eval_dirs  = cfg["eval_dir"]
        self.test_dirs  = cfg["test_dir"]
        for p in _collect_paths(self.train_dirs) + _collect_paths(self.eval_dirs) + _collect_paths(self.test_dirs):
            os.makedirs(p, exist_ok=True)

        # noise param
        noise = cfg["noise"]
        self.save_dir = noise["save_dir"]
        self.src_scale = noise["src_scale"]
        self.tgt_scale = noise["tgt_scale"]
        self.src_modulation = noise["src_modulation"]
        self.tgt_modulation = noise["tgt_modulation"]
        self.src_iter = noise["src_iter"]
        self.tgt_iter = noise["tgt_iter"]

        # preprocess param
        pp = cfg["preprocess"]
        self.random_seed = pp["random_seed"]
        self.sr = pp["sr"]
        self.nsample = pp["nsample"]
        self.n_fft = pp["n_fft"]
        self.hop = pp["hop_length"]
        self.window = pp["window"]
        self.nsplit = pp["nsplit"]

        # utils
        self.segmenter = Segmenter(self.nsample)
        self.baseline = BaselineFitter()
        self.noise_data = NoiseData(self.save_dir)

    def extract_noise(self):
        """Extract noise from test_dir and calculate PSD"""
        test_dir = self.orig_test
        csv_paths = [os.path.join(test_dir, f) for f in os.listdir(test_dir) if f.lower().endswith(".csv")]

        noise_chunks = []
        for csv in csv_paths:
            rows = read_csv(csv, skip_header=True)
            col = np.asarray(read_second_column(rows), dtype=np.float32)
            if len(col) >= 1024:
                noise_chunks.append(col[:1024])
            else:
                noise_chunks.append(col)
        if not noise_chunks:
            raise ValueError("cannot find csv from test_dir to extract noise")
        
        merged_noise = np.concatenate(noise_chunks)
        mu, sigma = self.noise_data.data_stats(merged_noise)
        print(f"Gaussian: mu={mu:.6f}, sigma={sigma:.6f}")

        f, psd = self.noise_data.data_psd(merged_noise, self.sr, plot=True)
        return f, psd

    def _add_noise(self, signal: np.ndarray, scale: float, modulation: float, f_base: np.ndarray, psd_base: np.ndarray, seed: int):
        """PSD interpolation to length followed by noise synthesis"""
        psd_fit = self.noise_data.psd_interpolate(f_base, psd_base, data=signal, sr=self.sr)  # noise psd interpolate
        f_fit = np.linspace(0, self.sr/2, len(psd_fit))

        noise = self.noise_data.noise_amp_modulation(out_len=len(signal), f=f_fit, psd=psd_fit, sr=self.sr, scale=scale, modulation=modulation, seed=seed)
        return signal + noise

    def _sig2spec(self, sig: np.ndarray):
        """Calculate STFT"""
        spec = librosa.stft(sig, n_fft=self.n_fft,
                            hop_length=self.hop, window=self.window)
        return librosa.amplitude_to_db(np.abs(spec), ref=np.max)

    def _save_png(self, spec: np.ndarray, save_path: str):
        """Save PNG"""
        plt.figure(figsize=(3, 3), dpi=100)
        librosa.display.specshow(spec, sr=self.sr,
                                 hop_length=self.hop, cmap='magma',
                                 x_axis=None, y_axis=None)
        plt.axis('off')
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
        plt.close()

    def _process_csv(self, csv_path: str, domain: str, scale: float, modulation: float, n_iter: int, f: np.ndarray, psd: np.ndarray): 
        """Processing CSV File"""
        rows = read_csv(csv_path, skip_header=True)
        raw = np.asarray(read_second_column(rows), dtype=np.float32)
        if raw.size == 0:
            return []
        
        fname = os.path.splitext(os.path.basename(csv_path))[0]
        specs = []

        for it in range(1, n_iter+1):
            noisy = self._add_noise(raw, scale, modulation, f, psd, seed=self.random_seed+it)  # noise add
            noisy = self.baseline.fit_correct(noisy)  # baseline

            for s_idx, (st, ed) in enumerate(self.segmenter.data_overlap(len(noisy))):
                seg = self.baseline.fit_correct(noisy[st:ed])  # baseline
                seg = self.segmenter.data_hanning(seg)  # hanning

                spec = self._sig2spec(seg)  # STFT
                tag = f"{fname}-A{it}-{s_idx}"
                specs.append((spec, tag, domain))  # list[tuple[np.ndarray, str, str]]:  # (spec, tag, domain)

        return specs

    def _process_dir(self, dir_key: str, domain: str, scale: float, modulation: float, n_iter: int, f: np.ndarray, psd: np.ndarray):
        """Processing CSV Files in Directory"""
        csv_dir_map = {
            "normal_dir": self.orig_normal,
            "s1_dir": self.orig_s1,
            "s2_dir": self.orig_s2,
            "test_dir":   self.orig_test,
        }
        csv_dir = csv_dir_map.get(dir_key)
        if not csv_dir or not os.path.isdir(csv_dir):
            return []
        
        out = []  # list[tuple[np.ndarray, str, str]]:
        for fn in os.listdir(csv_dir):
            if fn.lower().endswith(".csv"):
                out.extend(self._process_csv(csv_path=os.path.join(csv_dir, fn), domain=domain, 
                                             scale=scale, modulation=modulation, n_iter=n_iter, f=f, psd=psd))

        return out

    def _split_and_save(self, src_specs, tgt_specs):
        """Segment and save data"""
        bucket = defaultdict(list)
        for spec, tag, dom in src_specs:
            bucket[dom].append((spec, tag))
        
        # domain
        normals = bucket["normal"]
        s1s     = bucket["s1"]
        s2s     = bucket["s2"]

        # normal to train, eval01
        if isinstance(self.nsplit, float):  # ratio
            train_norm, eval_norm = train_test_split(normals, test_size=self.nsplit, random_state=self.random_seed, shuffle=True)
        else:  # absolute
            k = min(self.nsplit, len(normals))
            train_norm, eval_norm = train_test_split(normals, test_size=k, random_state=self.random_seed, shuffle=True)
        
        # eval02, eval03: sampling train_norm as s1·s2
        def pick_normals(src_pool, k):
            if k <= 0 or not src_pool:
                return []
            if k <= len(src_pool):
                return random.sample(src_pool, k)
            return random.choices(src_pool, k=k)

        eval02_norm = pick_normals(eval_norm, len(s1s))
        eval03_norm = pick_normals(eval_norm, len(s2s))

        # test00, test18, test23
        test00, test18, test23 = [], [], []
        for spec, tag, _ in tgt_specs:
            parts = tag.split("-")
            if len(parts) >= 3 and parts[2] == "18":
                test18.append((spec, tag))
            elif len(parts) >= 3 and parts[2] == "23":
                test23.append((spec, tag))
            else:
                test00.append((spec, tag))
        test00.extend(test18 + test23)

        # save helper
        def dump(specs, root, prefix):
            os.makedirs(root, exist_ok=True)
            for spec, tag in specs:
                self._save_png(spec, os.path.join(root, f"{prefix}-{tag}.png"))

        # save
        dump(train_norm, self.train_dirs["train"], "normal")

        dump(eval_norm, self.eval_dirs["eval01"], "normal")
        dump(eval02_norm, self.eval_dirs["eval02"], "normal")
        dump(eval03_norm, self.eval_dirs["eval03"], "normal")
        dump(s1s, self.eval_dirs["eval02"], "s1")
        dump(s2s, self.eval_dirs["eval03"], "s2")

        dump(test00, self.test_dirs["test00"], "test00")
        dump(test18, self.test_dirs["test18"], "test18")
        dump(test23, self.test_dirs["test23"], "test23")

    def run(self):
        """Run preprocessing pipeline"""
        # extract noise PSD
        f_base, psd_base = self.extract_noise()
        print(f"[1/6] noise data processed. ✅")

        # augmentation & STFT
        src_specs = []
        for step_idx, k in enumerate(('normal_dir', 's1_dir', 's2_dir'), start=2):
            domain = k.split('_')[0]
            src_specs += self._process_dir(dir_key=k, domain=domain, scale=self.src_scale, modulation=self.src_modulation, n_iter=self.src_iter, f=f_base, psd=psd_base)
            print(f"[{step_idx}/6] SRC {k} data processed. ✅")

        tgt_specs = self._process_dir(dir_key='test_dir', domain='test', scale=self.tgt_scale, modulation=self.tgt_modulation, n_iter=self.tgt_iter, f=f_base, psd=psd_base)
        print(f"[5/6] TGT test_dir data processed. ✅")

        # split → save png
        self._split_and_save(src_specs, tgt_specs)
        print(f"[6/6] SRC {len(src_specs)} seg, TGT {len(tgt_specs)} seg processed. ✅")


if __name__ == "__main__":
    cfg = get_configs()
    Preprocessor(cfg).run()
