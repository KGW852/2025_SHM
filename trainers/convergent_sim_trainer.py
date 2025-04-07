# trainers/convergent_sim_trainer.py

import torch
import mlflow

from models.convergent_sim import SimEncoder
from models.criterions.simsiam_loss import SimSiamLoss
from utils.model_utils import ModelUtils
from utils.logger import MLFlowLogger
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

        # optimizer, scheduler
        self.optimizer = Optimizer(self.cfg).get_optimizer(self.encoder.parameters())
        self.scheduler = Scheduler(self.cfg).get_scheduler(self.optimizer)

        # model utils: name, save manage
        self.model_utils = ModelUtils(self.cfg)

        # early stopper
        self.early_stopper = EarlyStopper(self.cfg).get_early_stopper()

        # params
        self.epochs = cfg["epochs"]
        self.log_every = cfg.get("log_every", 1)
        self.save_every = cfg.get("save_every", 1)

        # 

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
            loss = sim_loss

            # backprop
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        # scheduler step
        if self.scheduler is not None:
            self.scheduler.step()

        avg_loss = total_loss / len(train_loader)
        print(f"[Epoch {epoch+1}] simsiam_loss: {avg_loss:.4f}")

        return avg_loss

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

        return avg_loss

    def save_checkpoint(self, epoch: int):
        file_name = self.model_utils.get_model_name(epoch + 1)
        ckpt_path = self.model_utils.get_model_path(file_name)

        torch.save({
            'epoch': epoch + 1,
            'encoder_state_dict': self.encoder.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }, ckpt_path)
        
        print(f"Checkpoint saved to {ckpt_path}")

    def run(self, exp_class, train_loader, eval_loader=None):
        for epoch in range(self.epochs):
            train_loss = self.train_one_epoch(train_loader, epoch)  # train

            val_loss = None
            if eval_loader is not None and (epoch + 1) % self.log_every == 0:
                val_loss = self.validate(eval_loader, epoch)  # eval

            if val_loss is not None and self.early_stopper is not None:
                if self.early_stopper.step(val_loss):  # early stopping check(use val loss)
                    print("Early stopping triggered. Stop training.")
                    break

            # checkpoint
            if (epoch + 1) % self.save_every == 0:
                self.save_checkpoint(epoch)

