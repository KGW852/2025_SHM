# results/train_epoch_loss.py

import os
import pandas as pd
import matplotlib.pyplot as plt


plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "legend.fontsize": 10,
    "figure.dpi": 300,
    "axes.linewidth": 0.6,
    "lines.linewidth": 1.8,
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
    side = 4.0
    fig, axes = plt.subplots(1, 3, figsize=(side * 3, side), constrained_layout=True)

    for ax, (stem, title) in zip(axes, configs):
        train_csv = os.path.join(csv_dir, f"train_{stem}_loss.csv")
        eval_csv  = os.path.join(csv_dir, f"eval_{stem}_loss.csv")

        # skip subplot if either file is missing
        if not (os.path.exists(train_csv) and os.path.exists(eval_csv)):
            ax.set_axis_off()
            ax.text(0.5, 0.5, f"{stem.upper()} files\nnot found", ha="center", va="center", fontsize=12,)
            continue

        train_df = _read_loss_csv(train_csv)
        eval_df  = _read_loss_csv(eval_csv)

        # column extract
        epoch_col = train_df.columns[0] if "epoch" in train_df.columns[0].lower() else train_df.columns[3]
        loss_col  = "value" if "value" in train_df.columns else train_df.columns[5]
        
        # plot
        ax.plot(train_df[epoch_col], train_df[loss_col], marker="o", markersize=4, linestyle="-", color="blue", label="Train Loss")
        ax.plot(eval_df [epoch_col], eval_df [loss_col], marker="x", markersize=4, linestyle="--", color="blue", label="Val Loss")

        ax.set_title(title)
        ax.set_xlabel("Epochs")
        ax.set_ylabel("Loss")
        ax.grid(True, linestyle="--", linewidth=.4)
        ax.set_box_aspect(1)
        ax.legend(frameon=False, loc="best")

    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[✓] Figure saved → {os.path.relpath(save_path)}")

if __name__ == "__main__":
    plot_epoch_losses(csv_dir='./results/train epoch loss/csv/v3.2.4', save_dir='./results/train epoch loss/figure/v3.2.4')