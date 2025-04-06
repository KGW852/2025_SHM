# trainers/convergent_sim_trainer.py

import os
import torch
import torch.optim as optim

from models.convergent_sim import SimEncoder
from models.criterions.simsiam_loss import SimSiamLoss
from utils.model_utils import ModelUtils
from .tools import Optimizer, Scheduler, EarlyStopper


class ConvergentSimTrainer:
    """
    SimSiam Domain Adaptation trainer
    Args:
        cfg (dict): config dictionary
        device (torch.device): cuda or cpu
    """
    def __init__(self, cfg: dict, device: torch.device = None):
        self.cfg = cfg
        self.device = device if device is not None else torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )

        # model: MLPEncoder + SimSiam
        self.encoder = SimEncoder(
            in_dim=cfg["mlp"]["in_dim"],
            hidden_dims=cfg["mlp"]["in_hidden_dims"],
            latent_dim=cfg["mlp"]["latent_dim"],
            dropout=cfg["mlp"]["dropout"],
            proj_hidden_dim=cfg["sim"]["proj_hidden_dim"],
            proj_out_dim=cfg["sim"]["proj_out_dim"],
            pred_hidden_dim=cfg["sim"]["pred_hidden_dim"],
            pred_out_dim=cfg["sim"]["pred_out_dim"]
        ).to(self.device)

        # criterion
        self.simsiam_criterion = SimSiamLoss().to(self.device)

        # optimizer
        self.optimizer = optim.Adam(
            self.encoder.parameters(),
            lr=float(cfg["learning_rate"]),
            weight_decay=float(cfg["weight_decay"])
        )

        # (optional) scheduler
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=cfg["scheduler"]["args"]["step_size"],
            gamma=cfg["scheduler"]["args"]["gamma"]
        )

        # params
        self.epochs = cfg["epochs"]
        # self.lambda_sim = cfg["train"].get("lambda_sim", 1.0)

    def train_one_epoch(self, train_loader, epoch: int):
        self.encoder.train()

        total_loss = 0.0

        for batch_idx, data in enumerate(train_loader):
            (x_s, y_s), (x_t, y_t) = data  # x_s, x_t: (B, C, T)
            x_s = x_s.to(self.device)
            y_s = y_s.to(self.device)
            x_t = x_t.to(self.device)
            y_t = y_t.to(self.device)

            self.optimizer.zero_grad()

            # forward
            e_s, e_t, z_s, p_s, z_t, p_t = self.encoder(x_s, x_t)

            # loss
            sim_loss = self.simsiam_criterion(p_s, z_t, p_t, z_s)
            # loss = self.lambda_sim * sim_loss
            loss = sim_loss

            # backprop
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        # scheduler step
        self.scheduler.step()

        avg_loss = total_loss / len(train_loader)
        print(f"[Epoch {epoch+1}] simsiam_loss: {avg_loss:.4f}")

    def validate(self, eval_loader, epoch: int):
        self.encoder.eval()

        total_loss = 0.0
        with torch.no_grad():
            for batch_idx, data in enumerate(eval_loader):
                (x_s, y_s), (x_t, y_t) = data
                x_s = x_s.to(self.device)
                y_s = y_s.to(self.device)
                x_t = x_t.to(self.device)
                y_t = y_t.to(self.device)

                e_s, e_t, z_s, p_s, z_t, p_t = self.encoder(x_s, x_t)
                sim_loss = self.simsiam_criterion(p_s, z_t, p_t, z_s)
                total_loss += sim_loss.item()

        avg_loss = total_loss / len(eval_loader)
        print(f"[Val Epoch {epoch+1}] simsiam_loss: {avg_loss:.4f}")

    def save_checkpoint(self, epoch: int):
        save_path = os.path.join(self.cfg["mlflow_log_dir"], f"checkpoint_{epoch+1}.pth")
        torch.save({
            'epoch': epoch + 1,
            'encoder_state_dict': self.encoder.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }, save_path)
        print(f"Checkpoint saved to {save_path}")

    def run(self, train_loader, eval_loader=None):
        for epoch in range(self.epochs):
            self.train_one_epoch(train_loader, epoch)

            if eval_loader is not None and (epoch + 1) % self.cfg["log_every"] == 0:
                self.validate(eval_loader, epoch)

            # checkpoint
            if (epoch + 1) % self.cfg["save_every"] == 0:
                self.save_checkpoint(epoch)

