# trainers/mlp_ae_trainer.py

import torch
from tqdm import tqdm

from models.mlp_ae import MLPAE
from models.criterions.recon_loss import ReconLoss

from utils.model_utils import ModelUtils
from utils.logger import MLFlowLogger
from .tools import Optimizer, Scheduler, EarlyStopper


class MLPAETrainer:
    """
    Simple MLP AutoEncoder trainer
    Args:
        cfg (dict): config dictionary
        mlflow_logger (MLFlowLogger): MLflow logger(define param in 'main.py')
        device (torch.device): cuda or cpu
    """
    def __init__(self, cfg: dict, mlflow_logger: MLFlowLogger, device: torch.device = None):
        self.cfg = cfg
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # model: ConvergentAE(MLPEncoder + MLPDecoder)
        self.model = MLPAE(
            enc_in_dim=cfg["mlp"]["in_dim"],
            enc_hidden_dims=cfg["mlp"]["enc_hidden_dims"],
            enc_latent_dim=cfg["mlp"]["enc_latent_dim"],
            dec_latent_dim=cfg["mlp"]["dec_latent_dim"],
            dec_hidden_dims=cfg["mlp"]["dec_hidden_dims"],
            dec_out_channels=cfg["mlp"]["out_channels"],
            dec_out_seq_len=cfg["mlp"]["out_seq_len"],
            ae_dropout=cfg["mlp"]["dropout"],
            ae_use_batchnorm=cfg["mlp"]["use_batch_norm"]
        ).to(self.device)

        # loss, weight param
        self.recon_type = cfg["ae"].get("recon_type", "mae")
        self.ae_reduction = cfg["ae"].get("reduction", "mean")

        # criterion
        self.recon_criterion = ReconLoss(loss_type=self.recon_type, reduction=self.ae_reduction)

        # optimizer, scheduler, early stopper
        self.optimizer = Optimizer(self.cfg).get_optimizer(self.model.parameters())
        self.scheduler = Scheduler(self.cfg).get_scheduler(self.optimizer)
        self.early_stopper = EarlyStopper(self.cfg).get_early_stopper()

        # utils: model manage, mlflow
        self.model_utils = ModelUtils(self.cfg)
        self.mlflow_logger = mlflow_logger
        self.epochs = cfg["epochs"]
        self.log_every = cfg.get("log_every", 1)
        self.save_every = cfg.get("save_every", 1)

        # run_name: model_name
        self.run_name = self.model_utils.get_model_name()

    def train_epoch(self, train_loader, epoch: int):
        do_train = (epoch > 0)
        if do_train:
            self.model.train()
        else:
            self.model.eval()
        total_loss = 0.0
        recons_loss = 0.0
        recons_loss_s = 0.0
        recons_loss_t = 0.0

        pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch [{epoch}/{self.epochs}] Train", leave=False)
        for batch_idx, data in pbar:
            (x_s, y_s, _), (x_t, y_t, _) = data  # x_s, x_t: (B, C, T)
            x_s = x_s.to(self.device)
            y_s = y_s.to(self.device)
            x_t = x_t.to(self.device)
            y_t = y_t.to(self.device)
            
            self.optimizer.zero_grad()

            # forward
            (e_s, e_t, x_s_recon, x_t_recon) = self.model(x_s, x_t)

            # loss
            recon_loss_s = self.recon_criterion(x_s_recon, x_s)  # (pred, target)
            recon_loss_t = self.recon_criterion(x_t_recon, x_t)
            recon_loss = 0.5 * (recon_loss_s + recon_loss_t)

            loss = recon_loss

            # backprop
            if do_train:
                loss.backward()
                self.optimizer.step()

            # stats
            total_loss += loss.item()
            recons_loss += recon_loss.item()
            recons_loss_s += recon_loss_s.item()
            recons_loss_t += recon_loss_t.item()
            
            # mlflow log: global step
            if self.mlflow_logger is not None and batch_idx % 67 == 0:
                global_step = epoch * len(train_loader) + batch_idx
                self.mlflow_logger.log_metrics({"train_loss_step": loss.item(),"train_recon_loss_step": recon_loss.item(), 
                                                "train_recon_s_loss_step": recon_loss_s.item(), "train_recon_t_loss_step": recon_loss_t.item(), }, step=global_step)

            # tqdm
            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "recon": f"{recon_loss.item():.4f}",
                "recon_s": f"{recon_loss_s.item():.4f}",
                "recon_t": f"{recon_loss_t.item():.4f}"
                })
        
        # scheduler
        if do_train and (self.scheduler is not None):
            self.scheduler.step()
        
        # calc avg
        num_batches = len(train_loader)
        avg_loss = total_loss / num_batches
        avg_recon_loss = recons_loss / num_batches
        avg_recon_loss_s = recons_loss_s / num_batches
        avg_recon_loss_t = recons_loss_t / num_batches

        print(f"[Train] [Epoch {epoch}/{self.epochs}] "
              f"Avg: {avg_loss:.4f} | Recon: {avg_recon_loss:.4f} | "
              f"Recon_S: {avg_recon_loss_s:.4f} | Recon_T: {avg_recon_loss_t:.4f}")

        if self.mlflow_logger is not None:
            self.mlflow_logger.log_metrics({
                "train_loss": avg_loss,
                "train_recon_loss": avg_recon_loss,
                "train_recon_loss_s": avg_recon_loss_s,
                "train_recon_loss_t": avg_recon_loss_t
            }, step=epoch)
            
        return (avg_loss, avg_recon_loss)
    
    def eval_epoch(self, eval_loader, epoch: int):
        self.model.eval()
        total_loss = 0.0
        recons_loss = 0.0
        recons_loss_s = 0.0
        recons_loss_t = 0.0

        pbar = tqdm(enumerate(eval_loader), total=len(eval_loader), desc=f"Epoch [{epoch}/{self.epochs}] Eval", leave=False)
        with torch.no_grad():
            for batch_idx, data in pbar:
                (x_s, y_s, _), (x_t, y_t, _) = data
                x_s = x_s.to(self.device)
                y_s = y_s.to(self.device)
                x_t = x_t.to(self.device)
                y_t = y_t.to(self.device)

                # forward
                (e_s, e_t, x_s_recon, x_t_recon) = self.model(x_s, x_t)
                
                # loss
                recon_loss_s = self.recon_criterion(x_s_recon, x_s)  # (pred, target)
                recon_loss_t = self.recon_criterion(x_t_recon, x_t)
                recon_loss = 0.5 * (recon_loss_s + recon_loss_t)

                loss = recon_loss

                # stats
                total_loss += loss.item()
                recons_loss += recon_loss.item()
                recons_loss_s += recon_loss_s.item()
                recons_loss_t += recon_loss_t.item()

                # mlflow log: global step
                if self.mlflow_logger is not None and batch_idx % 1 == 0:
                    global_step = epoch * len(eval_loader) + batch_idx
                    self.mlflow_logger.log_metrics({"eval_loss_step": loss.item(), "eval_recon_loss_step": recon_loss.item(), 
                                                    "eval_recon_s_loss_step": recon_loss_s.item(), "eval_recon_t_loss_step": recon_loss_t.item(), }, step=global_step)
                
                # tqdm
                pbar.set_postfix({
                    "loss": f"{loss.item():.4f}",
                    "recon": f"{recon_loss.item():.4f}",
                    "recon_s": f"{recon_loss_s.item():.4f}",
                    "recon_t": f"{recon_loss_t.item():.4f}"
                    })

        # calc avg
        num_batches = len(eval_loader)
        avg_loss = total_loss / num_batches
        avg_recon_loss = recons_loss / num_batches
        avg_recon_loss_s = recons_loss_s / num_batches
        avg_recon_loss_t = recons_loss_t / num_batches

        print(f"[Eval]  [Epoch {epoch}/{self.epochs}] "
              f"Avg: {avg_loss:.4f} | Recon: {avg_recon_loss:.4f} | "
              f"Recon_S: {avg_recon_loss_s:.4f} | Recon_T: {avg_recon_loss_t:.4f}")
        
        if self.mlflow_logger is not None:
            self.mlflow_logger.log_metrics({
                "eval_loss": avg_loss,
                "eval_recon_loss": avg_recon_loss,
                "eval_recon_loss_s": avg_recon_loss_s,
                "eval_recon_loss_t": avg_recon_loss_t
            }, step=epoch)

        return (avg_loss, avg_recon_loss)

    def save_checkpoint(self, epoch: int):
        file_name = self.model_utils.get_file_name(epoch)
        ckpt_path = self.model_utils.get_file_path(file_name)

        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }, ckpt_path)
        
        print(f"Checkpoint saved: {ckpt_path}")

        # mlflow artifact upload
        if self.mlflow_logger is not None:
            self.mlflow_logger.log_artifact(ckpt_path, artifact_path="checkpoints")

    def run(self, train_loader, eval_loader=None, log_params_dict=None):
        """
        Training loop
        Args:
            train_loader, eval_loader
            log_params_dict (dict, optional): experiment parameters from main.py(expN)
        """
        if self.mlflow_logger is not None:
            self.mlflow_logger.start_run(run_name=self.run_name)
            if log_params_dict is not None:
                self.mlflow_logger.log_params(log_params_dict)

        for epoch in range(self.epochs + 1):
            train_loss_tuple = self.train_epoch(train_loader, epoch)  # train
            (train_avg, train_recon) = train_loss_tuple

            eval_loss_tuple = None
            if eval_loader is not None and epoch % self.log_every == 0:
                eval_loss_tuple = self.eval_epoch(eval_loader, epoch)  # eval
                (eval_avg, eval_recon) = eval_loss_tuple

            if eval_loss_tuple is not None and self.early_stopper is not None:  # early stopping check(use val loss)
                if self.early_stopper.step(eval_loss_tuple[0]):
                    print(f"Early stopping triggered at epoch {epoch}.")
                    self.save_checkpoint(epoch)
                    break

            # checkpoint
            if (epoch % self.save_every) == 0:
                self.save_checkpoint(epoch)

        if self.mlflow_logger is not None:
            run_id_path = self.model_utils.get_file_path(f"{self.run_name}.json")
            self.mlflow_logger.save_run_id(run_id_path)

            self.mlflow_logger.end_run()