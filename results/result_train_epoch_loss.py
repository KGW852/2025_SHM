# results/result_train_epoch_loss.py

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

def _read_loss_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "value" in df.columns:
        df["value"] = pd.to_numeric(df["value"].astype(str).str.replace("'", ""), errors="coerce")
    return df

def plot_epoch_losses(csv_dir, save_dir):
    fig_name = 'train_eval_epoch_loss.png'
    save_path = os.path.join(save_dir, fig_name)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # (filename stem, subplot title) in the desired left-to-right order
    configs = [("recon", "(a) Reconstruction"), 
               ("simsiam", "(b) CDA"), 
               ("svdd", "(c) AD Distance")]

    # subplot
    fig, axes = plt.subplots(3, 1, figsize=plt.rcParams["figure.figsize"])

    for ax, (stem, title) in zip(axes, configs):
        train_csv = os.path.join(csv_dir, f"train_{stem}_loss.csv")
        eval_csv  = os.path.join(csv_dir, f"eval_{stem}_loss.csv")

        # skip subplot if either file is missing
        if not (os.path.exists(train_csv) and os.path.exists(eval_csv)):
            ax.set_axis_off()
            ax.text(0.5, 0.5, f"{stem.upper()} files\nnot found", ha="center", va="center")
            continue

        train_df = _read_loss_csv(train_csv)
        eval_df  = _read_loss_csv(eval_csv)

        # column extract
        epoch_col = train_df.columns[0] if "epoch" in train_df.columns[0].lower() else train_df.columns[3]
        loss_col  = "value" if "value" in train_df.columns else train_df.columns[5]
        
        # plot
        ax.plot(train_df[epoch_col], train_df[loss_col], marker="o", linestyle="-", color="blue", label="Train Loss")
        ax.plot(eval_df [epoch_col], eval_df [loss_col], marker="x", linestyle="--", color="blue", label="Val Loss")

        ax.set_xlabel("Epochs", loc='right', labelpad=0)
        ax.set_ylabel("Loss")
        ax.grid(True, which='both', linestyle='--')
        ax.tick_params(axis='x', pad=0)
        ax.legend(frameon=False, loc="best")
        ax.set_title(title, loc='center', pad=0, y=-0.25)

    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[✓] Figure saved → {os.path.relpath(save_path)}")

if __name__ == "__main__":
    plot_epoch_losses(csv_dir='./results/train epoch loss/csv/v3.2.4', save_dir='./results/train epoch loss/figure/v3.2.4')