# trainers/convergent_sim_trainer.py

import torch
from tqdm import tqdm

from models.convergent_sim import ConvergentSim
from models.criterions.simsiam_loss import SimSiamLoss
from models.criterions.svdd_loss import DeepSVDDLoss

from utils.model_utils import ModelUtils
from utils.logger import MLFlowLogger
from .tools import Optimizer, Scheduler, EarlyStopper
from evaluation.modules.umap import plot_latent_alignment


class ConvergentSimTrainer:
    """
    SimSiam Domain Adaptation trainer
    Args:
        cfg (dict): config dictionary
        mlflow_logger (MLFlowLogger): MLflow logger(define param in 'main.py')
        device (torch.device): cuda or cpu
    """
    def __init__(self, cfg: dict, mlflow_logger: MLFlowLogger, device: torch.device = None):
        self.cfg = cfg
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # model: ConvergentSim(MLPEncoder + SimSiam + DeepSVDD)
        self.encoder = ConvergentSim(
            enc_in_dim=cfg["mlp"]["in_dim"],
            enc_hidden_dims=cfg["mlp"]["in_hidden_dims"],
            enc_latent_dim=cfg["mlp"]["latent_dim"],
            dropout=cfg["mlp"]["dropout"],
            proj_hidden_dim=cfg["sim"]["proj_hidden_dim"],
            proj_out_dim=cfg["sim"]["proj_out_dim"],
            pred_hidden_dim=cfg["sim"]["pred_hidden_dim"],
            pred_out_dim=cfg["sim"]["pred_out_dim"],
            svdd_latent_dim=cfg["svdd"]["latent_dim"]
        ).to(self.device)

        # criterion
        self.simsiam_criterion = SimSiamLoss().to(self.device)
        self.svdd_criterion = DeepSVDDLoss(
            nu=cfg["svdd"].get("nu", 0.1),
            reduction=cfg["svdd"].get("reduction", "mean")
        ).to(self.device)

        # loss, weight param
        self.recon_loss = cfg["ae"].get("recon", "mae")
        self.simsiam_lamda = cfg["ae"].get("simsiam_lamda", 1.0)
        self.svdd_lambda = cfg["ae"].get("svdd_lambda", 1.0)

        # optimizer, scheduler, early stopper
        self.optimizer = Optimizer(self.cfg).get_optimizer(self.encoder.parameters())
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

    def init_center(self, data_loader, eps=1e-5):
        """
        Initialize the center using the mean of the actual data distribution before training
        Args:
            data_loader: DataLoader containing only normal data (or entire dataset)
            eps (float): Constant for correcting values that are too close to zero
        """
        print("[DeepSVDD] Initializing center with mean of dataset ...")
        self.encoder.eval()

        # center vector
        latent_dim = self.encoder.deep_svdd.latent_dim
        center_sum = torch.zeros(latent_dim, device=self.device)
        n_samples = 0

        pbar = tqdm(enumerate(data_loader), total=len(data_loader), desc=f"Epoch [Init/{self.epochs}] Center", leave=False)
        with torch.no_grad():
            for batch_idx, data in pbar:
                (x_s, _), (x_t, _) = data  # x_s, x_t: (B, C, T)
                x_s = x_s.to(self.device)
                x_t = x_t.to(self.device)

                # forward
                (e_s, e_t, z_s, p_s, z_t, p_t, svdd_feat_s, svdd_feat_t) = self.encoder(x_s, x_t)
                batch_feat = torch.cat([svdd_feat_s, svdd_feat_t], dim=0)  # (2B, latent_dim)
                center_sum += batch_feat.sum(dim=0)
                n_samples += batch_feat.size(0)

        center_mean = center_sum / (n_samples + eps)
        self.encoder.deep_svdd.center.data = center_mean
        print(f"[DeepSVDD] center initialized. (norm={center_mean.norm():.4f})")

    def train_epoch(self, train_loader, epoch: int):
        do_train = (epoch > 0)
        if do_train:
            self.encoder.train()
        else:
            self.encoder.eval()
        total_loss = 0.0
        simsiam_loss = 0.0
        deep_svdd_loss = 0.0
        deep_svdd_loss_s = 0.0
        deep_svdd_loss_t = 0.0
        #last_outputs = None

        pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch [{epoch}/{self.epochs}] Train", leave=False)
        for batch_idx, data in pbar:
            (x_s, y_s), (x_t, y_t) = data  # x_s, x_t: (B, C, T)
            x_s = x_s.to(self.device)
            y_s = y_s.to(self.device)
            x_t = x_t.to(self.device)
            y_t = y_t.to(self.device)
            
            self.optimizer.zero_grad()

            # forward
            (e_s, e_t, z_s, p_s, z_t, p_t, svdd_feat_s, svdd_feat_t) = self.encoder(x_s, x_t)

            # loss
            sim_loss = self.simsiam_criterion(p_s, z_t, p_t, z_s)

            center = self.encoder.deep_svdd.center  # shape: [latent_dim]
            radius = self.encoder.deep_svdd.radius  # shape: []

            svdd_loss_s = self.svdd_criterion(svdd_feat_s, center, radius)
            svdd_loss_t = self.svdd_criterion(svdd_feat_t, center, radius)
            svdd_loss = 0.5 * (svdd_loss_s + svdd_loss_t)

            loss = self.simsiam_lamda * sim_loss + self.svdd_lambda * svdd_loss

            # backprop
            if do_train:
                loss.backward()
                self.optimizer.step()

            # stats
            total_loss += loss.item()
            simsiam_loss += sim_loss.item()
            deep_svdd_loss += svdd_loss.item()
            deep_svdd_loss_s += svdd_loss_s.item()
            deep_svdd_loss_t += svdd_loss_t.item()
            
            # mlflow log: global step
            if self.mlflow_logger is not None and batch_idx % 10 == 0:
                global_step = epoch * len(train_loader) + batch_idx
                self.mlflow_logger.log_metrics({"train_loss_step": loss.item(), "train_simsiam_step": sim_loss.item(), "train_svdd_step": svdd_loss.item(), }, step=global_step)
                
            # tqdm
            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "simsiam": f"{sim_loss.item():.4f}",
                "svdd": f"{svdd_loss.item():.4f}",
                "svdd_s": f"{svdd_loss_s.item():.4f}",
                "svdd_t": f"{svdd_loss_t.item():.4f}"
                })
            
            # for returning last batch outputs
            #last_outputs = (e_s, e_t, z_s, p_s, z_t, p_t, svdd_feat_s, svdd_feat_t)
        
        # scheduler
        if do_train and (self.scheduler is not None):
            self.scheduler.step()
        
        # calc avg
        num_batches = len(train_loader)
        avg_loss = total_loss / num_batches
        avg_sim_loss = simsiam_loss / num_batches
        avg_svdd_loss = deep_svdd_loss / num_batches
        avg_svdd_loss_s = deep_svdd_loss_s / num_batches
        avg_svdd_loss_t = deep_svdd_loss_t / num_batches

        print(f"[Train] [Epoch {epoch}/{self.epochs}] "
              f"Avg: {avg_loss:.4f} | SimSiam: {avg_sim_loss:.4f} | SVDD: {avg_svdd_loss:.4f} | "
              f"SVDD_S: {avg_svdd_loss_s:.4f} | SVDD_T: {avg_svdd_loss_t:.4f}")

        return (avg_loss, avg_sim_loss, avg_svdd_loss, avg_svdd_loss_s, avg_svdd_loss_t)

    def eval_epoch(self, eval_loader, epoch: int):
        self.encoder.eval()
        total_loss = 0.0
        simsiam_loss = 0.0
        deep_svdd_loss = 0.0
        deep_svdd_loss_s = 0.0
        deep_svdd_loss_t = 0.0
        #last_outputs = None

        pbar = tqdm(enumerate(eval_loader), total=len(eval_loader), desc=f"Epoch [{epoch}/{self.epochs}] Eval", leave=False)
        with torch.no_grad():
            for batch_idx, data in pbar:
                (x_s, y_s), (x_t, y_t) = data
                x_s = x_s.to(self.device)
                y_s = y_s.to(self.device)
                x_t = x_t.to(self.device)
                y_t = y_t.to(self.device)

                # forward
                (e_s, e_t, z_s, p_s, z_t, p_t, svdd_feat_s, svdd_feat_t) = self.encoder(x_s, x_t)

                # loss
                sim_loss = self.simsiam_criterion(p_s, z_t, p_t, z_s)

                center = self.encoder.deep_svdd.center  # shape: [latent_dim]
                radius = self.encoder.deep_svdd.radius  # shape: []

                svdd_loss_s = self.svdd_criterion(svdd_feat_s, center, radius)
                svdd_loss_t = self.svdd_criterion(svdd_feat_t, center, radius)
                svdd_loss = 0.5 * (svdd_loss_s + svdd_loss_t)

                loss = self.simsiam_lamda * sim_loss + self.svdd_lambda * svdd_loss

                # stats
                total_loss += loss.item()
                simsiam_loss += sim_loss.item()
                deep_svdd_loss += svdd_loss.item()
                deep_svdd_loss_s += svdd_loss_s.item()
                deep_svdd_loss_t += svdd_loss_t.item()

                # mlflow log: global step
                if self.mlflow_logger is not None and batch_idx % 10 == 0:
                    global_step = epoch * len(eval_loader) + batch_idx
                    self.mlflow_logger.log_metrics({"eval_loss_step": loss.item(), "eval_simsiam_step": sim_loss.item(), "eval_svdd_step": svdd_loss.item(), }, step=global_step)
                    
                # tqdm
                pbar.set_postfix({
                    "loss": f"{loss.item():.4f}",
                    "simsiam": f"{sim_loss.item():.4f}",
                    "svdd": f"{svdd_loss.item():.4f}",
                    "svdd_s": f"{svdd_loss_s.item():.4f}",
                    "svdd_t": f"{svdd_loss_t.item():.4f}"
                    })
                
                # for returning last batch outputs
                #last_outputs = (e_s, e_t, z_s, p_s, z_t, p_t, svdd_feat_s, svdd_feat_t)

        # calc avg
        num_batches = len(eval_loader)
        avg_loss = total_loss / num_batches
        avg_sim_loss = simsiam_loss / num_batches
        avg_svdd_loss = deep_svdd_loss / num_batches
        avg_svdd_loss_s = deep_svdd_loss_s / num_batches
        avg_svdd_loss_t = deep_svdd_loss_t / num_batches

        print(f"[Eval]  [Epoch {epoch}/{self.epochs}] "
            f"Avg: {avg_loss:.4f} | SimSiam: {avg_sim_loss:.4f} | SVDD: {avg_svdd_loss:.4f} | "
            f"SVDD_S: {avg_svdd_loss_s:.4f} | SVDD_T: {avg_svdd_loss_t:.4f}")

        return (avg_loss, avg_sim_loss, avg_svdd_loss, avg_svdd_loss_s, avg_svdd_loss_t)

    def save_checkpoint(self, epoch: int):
        file_name = self.model_utils.get_file_name(epoch)
        ckpt_path = self.model_utils.get_file_path(file_name)

        torch.save({
            'epoch': epoch,
            'encoder_state_dict': self.encoder.state_dict(),
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
        
        last_saved_epoch = None

        self.init_center(train_loader, eps=1e-5)

        for epoch in range(self.epochs + 1):
            train_loss_tuple = self.train_epoch(train_loader, epoch)  # train

            eval_loss_tuple = None
            if eval_loader is not None and epoch % self.log_every == 0:
                eval_loss_tuple = self.eval_epoch(eval_loader, epoch)  # eval

            # mlflow metrics log
            if self.mlflow_logger is not None:
                (train_avg, train_sim, train_svdd, train_svdd_s, train_svdd_t) = train_loss_tuple
                metrics = {
                    "train_loss": train_avg,
                    "train_simsiam": train_sim,
                    "train_svdd": train_svdd,
                    "train_svdd_s": train_svdd_s,
                    "train_svdd_t": train_svdd_t,
                }
                if eval_loss_tuple is not None:
                    (eval_avg, eval_sim, eval_svdd, eval_svdd_s, eval_svdd_t) = eval_loss_tuple
                    metrics.update({
                        "eval_loss": eval_avg,
                        "eval_simsiam": eval_sim,
                        "eval_svdd": eval_svdd,
                        "eval_svdd_s": eval_svdd_s,
                        "eval_svdd_t": eval_svdd_t,
                    })
                self.mlflow_logger.log_metrics(metrics, step=epoch)

            if eval_loss_tuple is not None and self.early_stopper is not None:  # early stopping check(use val loss)
                if self.early_stopper.step(eval_loss_tuple[0]):
                    print(f"Early stopping triggered at epoch {epoch}.")
                    self.save_checkpoint(epoch)
                    last_saved_epoch = epoch
                    break

            # checkpoint
            if (epoch % self.save_every) == 0:
                self.save_checkpoint(epoch)
                last_saved_epoch = epoch

        # MLflow Registry final model and return run_id, last_saved_epoch
        """
        if self.mlflow_logger is not None and last_saved_epoch is not None:
            final_ckpt_path = self.model_utils.get_file_path(self.model_utils.get_file_name(last_saved_epoch))
            self.mlflow_logger.register_model(model_path=final_ckpt_path, model_name=self.run_name)
        """
        run_id = self.mlflow_logger.run_id if self.mlflow_logger is not None else None
        return run_id, last_saved_epoch
            
        """
        if self.mlflow_logger is not None:
            self.mlflow_logger.end_run()
        """