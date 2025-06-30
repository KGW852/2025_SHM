# results/result_compare_fem_exp_TH.py

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


WIDTH_MM   = 84
WIDTH_INCH = WIDTH_MM / 25.4
HEIGHT_IN  = 8.2

plt.rcParams.update({
    "figure.figsize": (WIDTH_INCH, HEIGHT_IN),
    "figure.dpi": 600,
    "savefig.dpi": 600,
    #"figure.autolayout": True,
    "figure.subplot.hspace": 0.10,
    #"figure.subplot.bottom": 0.00,

    "font.family": "Times New Roman",
    "font.size": 8,
    "axes.titlesize": 8,
    "axes.labelsize": 8,
    "legend.fontsize": 8,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,

    "axes.linewidth": 0.5,
    "lines.linewidth": 0.5,
    "lines.markersize": 4,
    "grid.alpha": 0.25,
})

# (filename, subplot title) in the desired order
MAP_CONFIGS = {
    'fem_orig': ('A-T-40-Down(Test)-12.csv', '(a) FEM Original TH'),
    'exp_orig': ('A-T-18-40-Down-4.csv', '(b) Experimental Original TH'),
    'fem_cut': ('A-T-40-Down(Test)-12-cut.csv', '(c) FEM Free-vib. TH'),
    'exp_cut': ('A-T-18-40-Down-4-cut.csv', '(d) Experimental Free-vib. TH'),
    'fem_fft': ('A-T-40-Down(Test)-12-cut_FFT.csv', '(e) FEM Free-vib. FFT'),
    'exp_fft': ('A-T-18-40-Down-4-cut_FFT.csv', '(f) Experimental Free-vib. FFT')
}

def _read_signal_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "value" in df.columns:
        df["value"] = pd.to_numeric(df["value"].astype(str).str.replace("'", ""), errors="coerce")
    return df

def plot_compare_signal(csv_dir, save_dir):
    fig_name = 'compare_fem_exp.png'
    save_path = os.path.join(save_dir, fig_name)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # subplot
    fig, axes = plt.subplots(nrows=6, ncols=1, figsize=plt.rcParams["figure.figsize"])

    for idx, (key, (fname, title)) in enumerate(MAP_CONFIGS.items()):
        ax = axes[idx]

        csv_path = os.path.join(csv_dir, fname)
        if not os.path.isfile(csv_path):
            raise FileNotFoundError(f"{csv_path} not found")
        df = _read_signal_csv(csv_path)
        
        # plot
        ax.plot(df.iloc[:, 0], df.iloc[:, 1], color='blue')
        ax.grid(True, which='both', linestyle='--')

        # vertical line
        if idx in (0, 1):
            ax.axvline(x=78.0, color='red', linestyle='--', linewidth=0.8)

        if 'fft' in key.lower():
            xlabel, ylabel = "Freq. (Hz)", "Magnitude"
        else:
            xlabel, ylabel = "Time (s)", "Amplitude"
    
        ax.set_xlabel(xlabel, loc='right', labelpad=0)
        ax.set_ylabel(ylabel)
        ax.tick_params(axis='x', pad=0)
        ax.set_title(title, loc='center', pad=0, y=-0.25)

    fig.tight_layout(pad=0.0, h_pad=0.6)
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[✓] Figure saved → {os.path.relpath(save_path)}")

if __name__ == "__main__":
    plot_compare_signal(csv_dir='./results/compare fem exp/csv', save_dir='./results/compare fem exp/figure')