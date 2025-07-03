# results/result_umap.py

import os
import pandas as pd
import matplotlib.pyplot as plt


WIDTH_MM = 78
HEIGHT_MM = 150
WIDTH_IN = WIDTH_MM / 25.4
HEIGHT_IN = HEIGHT_MM / 25.4

plt.rcParams.update({
    "figure.figsize": (WIDTH_IN, HEIGHT_IN),
    "figure.dpi": 600,
    "savefig.dpi": 600,

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
    'enc_only': ('v3.0_resnet_ae/umap_s2(18)_encoder_epoch0.csv', '(a)'),
    'ae_only': ('v3.0_resnet_ae/umap_s2(18)_encoder_epoch5.csv', '(b)'),
    'da_after': ('v3.2.4/umap_s2(18)_encoder_epoch30.csv', '(c)')
}
COLOR_MAP  = {0: '#000000', 2: '#d62728', 18: '#1f77b4'}
MARKER_MAP = {0: 'o', 2: 'x', 18: 's'}

LEGEND_MAP = {
    0 : "FEM (Normal)",
    18: "2018-measured (Normal)",
    2 : "FEM (Anomaly)",
}

def _read_umap_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "value" in df.columns:
        df["value"] = pd.to_numeric(df["value"].astype(str).str.replace("'", ""), errors="coerce")
    return df

def plot_umap(csv_dir, save_dir):
    fig_name  = 'umap_s2(18)_encoder.png'
    save_path = os.path.join(save_dir, fig_name)
    os.makedirs(save_dir, exist_ok=True)

    # subplot
    n_plots = len(MAP_CONFIGS)
    fig, axes = plt.subplots(n_plots, 1, figsize=plt.rcParams["figure.figsize"])

    for idx, (fname, title) in enumerate(MAP_CONFIGS.values()):
        ax = axes[idx]
        csv_path = os.path.join(csv_dir, fname)
        if not os.path.isfile(csv_path):
            raise FileNotFoundError(f"{csv_path} not found")
        
        df = _read_umap_csv(csv_path)
        class_ids = sorted(df.iloc[:, 2].unique())  # e.g. [0, 2, 18]

        for cid in class_ids:
            color, marker = COLOR_MAP[cid], MARKER_MAP[cid]
            sub           = df[df.iloc[:, 2] == cid]

            # legend
            legend = LEGEND_MAP.get(cid, f'class {cid}')

            # subplot scatter
            ax.scatter(
                    sub.iloc[:, 0], sub.iloc[:, 1], marker=marker, 
                    facecolors='none' if marker in ['o','s'] else color,
                    edgecolors=color, linewidths=0.4, s=6, label=legend
                )
        
        # subplot layour
        ax.legend(frameon=False)
        ax.grid(True, which='both', linestyle='--')
        ax.tick_params(axis='x', pad=0)
        ax.set_title(title, loc='center', pad=0, y=-0.15)
    
    # plot setting
    fig.tight_layout(pad=0.0, h_pad=0.75)
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure saved: {os.path.relpath(save_path)}")


if __name__ == "__main__":
    plot_umap(csv_dir='./results/umap/csv', save_dir='./results/umap/figure')