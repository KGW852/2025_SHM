# data/tools_preprocess.py

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy import signal, stats
from scipy.interpolate import interp1d
from math import ceil


# data segment and window
class Segmenter:
    def __init__(self, nsample):
        self.nsample = nsample

    # automatic overlap size indices: equal distribution
    def data_overlap(self, nlength: int):
        seg_size = self.nsample
        if nlength <= seg_size:
            return [(0, nlength)]
        
        count = (nlength - seg_size) / seg_size

        s = int(ceil(count)) + 1 # number of segment: count +1

        # min overlap: 50%
        min_s = int(ceil(2 * (nlength - seg_size) / seg_size)) + 1
        s = max(s, min_s)
        step = (nlength - seg_size) / (s - 1) # equal step size or distribution

        segments = []
        for i in range(s):
            start_idx = int(round(i * step))
            end_idx = start_idx + seg_size

            if i == s - 1:
                start_idx = nlength - seg_size
                end_idx = nlength

            segments.append((start_idx, end_idx))

        return segments

    # apply hanning window in segments data
    def data_hanning(self, data: np.ndarray) -> np.ndarray:
        length = len(data)
        hanning_win = np.hanning(length) # make hanning window with fit length
        
        return data * hanning_win
    
# Noise data porcessor
class NoiseData:
    def __init__(self, save_dir):
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)

    # Noise data load & merge
    def data_merge(self, file_dir):
        # Load all csv files
        data = []

        for file in os.listdir(file_dir):
            if not file.endswith('.csv'):
                continue
            file_path = os.path.join(file_dir, file)

            # csv read
            csv_data = pd.read_csv(file_path, header=0)
            signal = pd.to_numeric(csv_data.iloc[:, 0], errors='coerce').dropna().values.astype(np.float32)

            data.append(signal)

        # merge all data
        merge_data = np.concatenate(data)
        print(f"Total merged data size: {len(merge_data)}")

        return merge_data
    
    # Noise data analyze: distribution
    def data_stats(self, data, plot=True):
        # data distribution
        mu, sigma = stats.norm.fit(data)

        # Save
        save_path = os.path.join(self.save_dir, 'data stats.txt')
        with open(save_path, 'w') as f:
            f.write(f"mu: {mu:.6f}\n")
            f.write(f"sigma: {sigma:.6f}\n")
        print(f"Stats saved: {save_path}")
    
        # plot normal distribution
        if plot:
            plt.figure(figsize=(8, 5))
            plt.hist(data, bins=50, density=True, alpha=0.6, color='g', label="Histogram")
        
            xmin, xmax = plt.xlim()
            x = np.linspace(xmin, xmax, 100)
            p = stats.norm.pdf(x, mu, sigma)
            plt.plot(x, p, 'k', linewidth=2, label="Fitted Normal Distribution")
        
            plt.title("Data Distribution")
            plt.xlabel("Acc")
            plt.ylabel("Density")
            plt.legend()
            plt.grid()
        
            # Save
            save_path = os.path.join(self.save_dir, 'data normal distribution.png')
            plt.savefig(save_path, dpi=300)
            print(f"Plot saved: {save_path}")

        return mu, sigma

    # Noise data analyze: Power Spectral Density
    def data_psd(self, data, sr, plot=True):
        # Power Spectral Density
        length = min(512, len(data))
        f, psd = signal.welch(data, fs=sr, nperseg=length, noverlap=length//2)

        # plot PSD
        if plot:
            plt.figure(figsize=(8, 5))
            plt.semilogy(f, psd)
            plt.title("Power Spectral Density")
            plt.xlabel("Frequency [Hz]")
            plt.ylabel("Power/Frequency [dB/Hz]")
            plt.grid()
        
            # Save
            save_path = os.path.join(self.save_dir, 'noise data PSD.png')
            plt.savefig(save_path, dpi=300)
            print(f"Plot saved: {save_path}")

        return f, psd

    # Noise PSD interpolation
    def psd_interpolate(self, f, psd, data, sr):
        # data
        data_len = len(data)
        freq_len = data_len // 2 + 1

        # frequency interpolation
        f_new = np.linspace(0, sr/2, freq_len)

        # PSD interpolation
        f_interp = interp1d(f, psd, kind='linear', fill_value='extrapolate')
        interpolate_psd = f_interp(f_new)

        return interpolate_psd

    # Noise data generate: random phase
    def data_noise(self, psd, data, seed=None):
        if seed is not None:
            np.random.seed(seed)

        # amplitude spectrum
        amp_spectrum = np.sqrt(psd)

        # random phase
        nf = len(amp_spectrum)
        phase = 2 * np.pi * np.random.rand(nf)

        # complex spectrum
        comp_spectrum = amp_spectrum * np.exp(1j * phase)

        # Inverse FFT
        noise = np.fft.irfft(comp_spectrum, n=len(data))

        return noise

    # Noise data generation: random amplitude modulation
    def amplitude_modulation(self, signal, mod_freq, mod_index, sr, random_phase, seed):
        if seed is not None and random_phase:
            np.random.seed(seed)

        length = len(signal)
        time = np.arange(length) / sr
        phase = 0.0 # Fixed phase

        if random_phase:
            phase = 2 * np.pi * np.random.rand()

        # Modulation signal: Amp(t) = 1 + m * sin(2πf t + φ)
        modulator = 1.0 + mod_index * np.sin(2 * np.pi * mod_freq * time + phase)

        # Amplitude modulation
        signal_mod = signal * modulator

        return signal_mod
    
# Baseline fitting
class BaselineFitter:
    def __init__(self):
        self.coefficients = None

    def fit(self, y, x=None):
        if x is None:
            x = np.arange(len(y))
        self.coefficients = np.polyfit(x, y, 1)
        return self
    
    def get_baseline(self, x):
        if self.coefficients is None:
            raise ValueError("Must fit the model first.")
        return np.polyval(self.coefficients, x)
    
    def correct(self, y, x=None):
        if x is None:
            x = np.arange(len(y))
        baseline = self.get_baseline(x)
        return y - baseline
    
    def fit_correct(self, y, x=None):
        self.fit(y, x)
        return self.correct(y, x)
    
