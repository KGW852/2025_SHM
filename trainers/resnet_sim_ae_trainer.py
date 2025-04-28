# trainers/resnet_sim_ae_trainer.py

import torch
from tqdm import tqdm

from models.resnet_sim_ae import ResNetSimAE
from models.criterions.recon_loss import ReconLoss
from models.criterions.simsiam_loss import SimSiamLoss

from utils.model_utils import ModelUtils
from utils.logger import MLFlowLogger
from .tools import Optimizer, Scheduler, EarlyStopper


class ResSimAETrainer:
    """
    Trainer for ResNetAE model, handling image data for source and target domains.
    Args:
        cfg (dict): Configuration dictionary.
        mlflow_logger (MLFlowLogger): MLflow logger instance.
        device (torch.device): Device to use for training (cuda or cpu).
    """
    def __init__(self, cfg: dict, mlflow_logger: MLFlowLogger, device: torch.device = None):
        self.cfg = cfg
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # model: ResNetSimAE (ResNet50Encoder + Siamese + ImageDecoder)
        self.model = ResNetSimAE(
            channels=cfg["resnet_ae"].get("channels", 3),
            height=cfg["resnet_ae"].get("height"),
            width=cfg["resnet_ae"].get("width"),
            enc_freeze=cfg["resnet_ae"].get("enc_freeze", True),
            dec_latent_dim=cfg["resnet_ae"].get("dec_latent_dim", 2048),
            dec_hidden_dims=cfg["resnet_ae"].get("dec_hidden_dims"),
            dropout=cfg["resnet_ae"].get("dropout", 0.0),
            use_batchnorm=cfg["resnet_ae"].get("use_batch_norm", False),

            proj_hidden_dim=cfg["sim"].get("proj_hidden_dim", 512),
            proj_out_dim=cfg["sim"].get("proj_out_dim", 512),
            pred_hidden_dim=cfg["sim"].get("pred_hidden_dim", 256),
            pred_out_dim=cfg["sim"].get("pred_out_dim", 512),
        ).to(self.device)

        # loss, weight param
        self.recon_type = cfg["ae"].get("recon_type", "mae")
        self.ae_reduction = cfg["ae"].get("reduction", "mean")
        self.simsiam_lamda = cfg["ae"].get("simsiam_lamda", 1.0)

        # criterion
        self.recon_criterion = ReconLoss(loss_type=self.recon_type, reduction=self.ae_reduction)
        self.simsiam_criterion = SimSiamLoss().to(self.device)

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
        """
        Train the model for one epoch.
        """
        do_train = (epoch > 0)
        if do_train:
            self.model.train()
        else:
            self.model.eval()
        """
        # debuging code
        if epoch == 1:
            print("=== Checking requires_grad in layer4 ===")
            for name, param in self.model.encoder.layer4.named_parameters():
                print(f"{name}, requires_grad={param.requires_grad}")
        """
        total_loss = 0.0
        recons_loss = 0.0
        simsiam_loss = 0.0
        recons_loss_s = 0.0
        recons_loss_t = 0.0

        pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch [{epoch}/{self.epochs}] Train", leave=False)
        for batch_idx, data in pbar:
            (x_s, y_s, _), (x_t, y_t, _) = data  # x_s, x_t: (B, C, H, W)
            x_s = x_s.to(self.device)
            x_t = x_t.to(self.device)

            self.optimizer.zero_grad()

            # Forward pass
            (e_s, e_t, z_s, p_s, z_t, p_t, x_s_recon, x_t_recon) = self.model(x_s, x_t)

            # Compute reconstruction losses
            recon_loss_s = self.recon_criterion(x_s_recon, x_s)
            recon_loss_t = self.recon_criterion(x_t_recon, x_t)
            recon_loss = 0.5 * (recon_loss_s + recon_loss_t)
            sim_loss = self.simsiam_criterion(p_s, z_t, p_t, z_s)

            loss = recon_loss + self.simsiam_lamda * sim_loss

            # Backpropagation
            if do_train:
                loss.backward()
                self.optimizer.step()

            # Accumulate losses
            total_loss += loss.item()
            recons_loss += recon_loss.item()
            simsiam_loss += sim_loss.item()
            recons_loss_s += recon_loss_s.item()
            recons_loss_t += recon_loss_t.item()

            # Log to MLflow every 67 batches
            if self.mlflow_logger is not None and batch_idx % 67 == 0:
                global_step = epoch * len(train_loader) + batch_idx
                self.mlflow_logger.log_metrics({
                    "train_loss_step": loss.item(),
                    "train_recon_loss_step": recon_loss.item(),
                    "train_simsiam_loss_step": sim_loss.item(),
                }, step=global_step)

            # Update progress bar
            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "recon": f"{recon_loss.item():.4f}",
                "simsiam": f"{sim_loss.item():.4f}",
            })

        # Scheduler step
        if do_train and self.scheduler is not None:
            self.scheduler.step()

        # Calculate average losses
        num_batches = len(train_loader)
        avg_loss = total_loss / num_batches
        avg_recon_loss = recons_loss / num_batches
        avg_sim_loss = simsiam_loss / num_batches
        avg_recon_loss_s = recons_loss_s / num_batches
        avg_recon_loss_t = recons_loss_t / num_batches

        print(f"[Train] [Epoch {epoch}/{self.epochs}] "
              f"Avg: {avg_loss:.4f} | Recon: {avg_recon_loss:.4f} | SimSiam: {avg_sim_loss:.4f} | "
              f"Recon_S: {avg_recon_loss_s:.4f} | Recon_T: {avg_recon_loss_t:.4f}")

        if self.mlflow_logger is not None:
            self.mlflow_logger.log_metrics({
                "train_loss": avg_loss,
                "train_recon_loss": avg_recon_loss,
                "train_simsiam_loss": avg_sim_loss,
                "train_recon_loss_s": avg_recon_loss_s,
                "train_recon_loss_t": avg_recon_loss_t
            }, step=epoch)

        return (avg_loss, avg_recon_loss, avg_sim_loss)

    def eval_epoch(self, eval_loader, epoch: int):
        """
        Evaluate the model for one epoch.
        Args:
            eval_loader: DataLoader for evaluation data.
            epoch (int): Current epoch number.
        Returns:
            tuple: (avg_loss, avg_recon_loss)
        """
        self.model.eval()
        total_loss = 0.0
        recons_loss = 0.0
        simsiam_loss = 0.0
        recons_loss_s = 0.0
        recons_loss_t = 0.0

        pbar = tqdm(enumerate(eval_loader), total=len(eval_loader), desc=f"Epoch [{epoch}/{self.epochs}] Eval", leave=False)
        with torch.no_grad():
            for batch_idx, data in pbar:
                (x_s, y_s, _), (x_t, y_t, _) = data
                x_s = x_s.to(self.device)
                x_t = x_t.to(self.device)

                # Forward pass
                (e_s, e_t, z_s, p_s, z_t, p_t, x_s_recon, x_t_recon) = self.model(x_s, x_t)

                # Compute reconstruction losses
                recon_loss_s = self.recon_criterion(x_s_recon, x_s)
                recon_loss_t = self.recon_criterion(x_t_recon, x_t)
                recon_loss = 0.5 * (recon_loss_s + recon_loss_t)
                sim_loss = self.simsiam_criterion(p_s, z_t, p_t, z_s)

                loss = recon_loss + self.simsiam_lamda * sim_loss

                # Accumulate losses
                total_loss += loss.item()
                recons_loss += recon_loss.item()
                simsiam_loss += sim_loss.item()
                recons_loss_s += recon_loss_s.item()
                recons_loss_t += recon_loss_t.item()

                # Log to MLflow every batch
                if self.mlflow_logger is not None and batch_idx % 1 == 0:
                    global_step = epoch * len(eval_loader) + batch_idx
                    self.mlflow_logger.log_metrics({
                        "eval_loss_step": loss.item(),
                        "eval_recon_loss_step": recon_loss.item(),
                        "eval_simsiam_loss_step": sim_loss.item(),
                    }, step=global_step)

                # Update progress bar
                pbar.set_postfix({
                    "loss": f"{loss.item():.4f}",
                    "recon": f"{recon_loss.item():.4f}",
                    "simsiam": f"{sim_loss.item():.4f}",
                })

        # Calculate average losses
        num_batches = len(eval_loader)
        avg_loss = total_loss / num_batches
        avg_recon_loss = recons_loss / num_batches
        avg_sim_loss = simsiam_loss / num_batches
        avg_recon_loss_s = recons_loss_s / num_batches
        avg_recon_loss_t = recons_loss_t / num_batches

        print(f"[Eval]  [Epoch {epoch}/{self.epochs}] "
              f"Avg: {avg_loss:.4f} | Recon: {avg_recon_loss:.4f} | SimSiam: {avg_sim_loss:.4f} | "
              f"Recon_S: {avg_recon_loss_s:.4f} | Recon_T: {avg_recon_loss_t:.4f}")

        if self.mlflow_logger is not None:
            self.mlflow_logger.log_metrics({
                "eval_loss": avg_loss,
                "eval_recon_loss": avg_recon_loss,
                "eval_simsiam_loss": avg_sim_loss,
                "eval_recon_loss_s": avg_recon_loss_s,
                "eval_recon_loss_t": avg_recon_loss_t
            }, step=epoch)

        return (avg_loss, avg_recon_loss, avg_sim_loss)

    def save_checkpoint(self, epoch: int):
        """
        Save model checkpoint and log to MLflow.
        """
        file_name = self.model_utils.get_file_name(epoch)
        ckpt_path = self.model_utils.get_file_path(file_name)

        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }, ckpt_path)

        print(f"Checkpoint saved: {ckpt_path}")

        # Upload checkpoint to MLflow
        #if self.mlflow_logger is not None:
            #self.mlflow_logger.log_artifact(ckpt_path, artifact_path="checkpoints")

    def run(self, train_loader, eval_loader=None, log_params_dict=None):
        """
        Run the training loop.
        Args:
            train_loader: DataLoader for training data.
            eval_loader: DataLoader for evaluation data (optional).
            log_params_dict (dict, optional): Parameters to log in MLflow.
        """
        if self.mlflow_logger is not None:
            self.mlflow_logger.start_run(run_name=self.run_name)
            if log_params_dict is not None:
                self.mlflow_logger.log_params(log_params_dict)

        for epoch in range(self.epochs + 1):
            train_loss_tuple = self.train_epoch(train_loader, epoch)  # Train
            (train_avg, train_recon, train_sim) = train_loss_tuple

            eval_loss_tuple = None
            if eval_loader is not None and epoch % self.log_every == 0:
                eval_loss_tuple = self.eval_epoch(eval_loader, epoch)  # Evaluate
                (eval_avg, eval_recon, eval_sim) = eval_loss_tuple

            # Early stopping check using validation loss
            if eval_loss_tuple is not None and self.early_stopper is not None:
                if self.early_stopper.step(eval_loss_tuple[0]):
                    print(f"Early stopping triggered at epoch {epoch}.")
                    self.save_checkpoint(epoch)
                    break

            # Save checkpoint periodically
            if (epoch % self.save_every) == 0:
                self.save_checkpoint(epoch)

        if self.mlflow_logger is not None:
            run_id_path = self.model_utils.get_file_path(f"{self.run_name}.json")
            self.mlflow_logger.save_run_id(run_id_path)
            self.mlflow_logger.end_run()