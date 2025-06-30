# results/result_umap.py

import os
import pandas as pd
import matplotlib.pyplot as plt


WIDTH_MM = 80
HEIGHT_MM = 150
WIDTH_INCH = WIDTH_MM / 25.4
HEIGHT_IN = HEIGHT_MM / 25.4

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
    "lines.markersize": 2,
    "grid.alpha": 0.25,
})

# (filename, subplot title) in the desired order
MAP_CONFIGS = {
    'enc_only': ('v3.0_resnet_ae/umap_s2(18)_encoder_epoch0.csv', '(a) Before DA'),
    'ae_only': ('v3.0_resnet_ae/umap_s2(18)_encoder_epoch5.csv', '(b) Before DA'),
    'da_after': ('v3.2.4/umap_s2(18)_encoder_epoch30.csv', '(c) After DA')
}

def _read_umap_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "value" in df.columns:
        df["value"] = pd.to_numeric(df["value"].astype(str).str.replace("'", ""), errors="coerce")
    return df

def plot_umap(csv_dir, save_dir):
    fig_name = 'umap_s2(18)_encoder.png'
    save_path = os.path.join(save_dir, fig_name)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # subplot
    fig, axes = plt.subplots(3, 1, figsize=plt.rcParams["figure.figsize"])

    for idx, (key, (fname, title)) in enumerate(MAP_CONFIGS.items()):
        ax = axes[idx]

        csv_path = os.path.join(csv_dir, fname)
        if not os.path.isfile(csv_path):
            raise FileNotFoundError(f"{csv_path} not found")
        df = _read_umap_csv(csv_path)
        
        # plot
        ax.plot(df.iloc[:, 0], df.iloc[:, 1], color='blue')
        ax.grid(True, which='both', linestyle='--')
        ax.tick_params(axis='x', pad=0)
        ax.set_title(title, loc='center', pad=0, y=-0.25)

    fig.tight_layout(pad=0.0, h_pad=0.6)
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[✓] Figure saved → {os.path.relpath(save_path)}")


if __name__ == "__main__":
    plot_umap(csv_dir='./results/umap/csv', save_dir='./results/umap/figure')