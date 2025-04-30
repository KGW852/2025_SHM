# trainers/sim_svdd_ae_trainer.py

import torch
from tqdm import tqdm

from models.sim_svdd_ae import SimSVDDAE
from models.criterions.simsiam_loss import SimSiamLoss
from models.criterions.recon_loss import ReconLoss
from models.criterions.svdd_loss import DeepSVDDLoss

from utils.model_utils import ModelUtils
from utils.logger import MLFlowLogger
from .tools import Optimizer, Scheduler, EarlyStopper


class SimSVDDAETrainer:
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

        # model: ConvergentAE(MLPEncoder + SimSiam + DeepSVDD + MLPDecoder)
        self.model = SimSVDDAE(
            enc_in_dim=cfg["mlp"]["in_dim"],
            enc_hidden_dims=cfg["mlp"]["enc_hidden_dims"],
            enc_latent_dim=cfg["mlp"]["enc_latent_dim"],
            dec_latent_dim=cfg["mlp"]["dec_latent_dim"],
            dec_hidden_dims=cfg["mlp"]["dec_hidden_dims"],
            dec_out_channels=cfg["mlp"]["out_channels"],
            dec_out_seq_len=cfg["mlp"]["out_seq_len"],
            ae_dropout=cfg["mlp"]["dropout"],
            ae_use_batchnorm=cfg["mlp"]["use_batch_norm"],

            proj_hidden_dim=cfg["sim"]["proj_hidden_dim"],
            proj_out_dim=cfg["sim"]["proj_out_dim"],
            pred_hidden_dim=cfg["sim"]["pred_hidden_dim"],
            pred_out_dim=cfg["sim"]["pred_out_dim"],

            svdd_in_dim=cfg["svdd"]["in_dim"],
            svdd_hidden_dims=cfg["svdd"]["hidden_dims"],
            svdd_latent_dim=cfg["svdd"]["latent_dim"],
            svdd_center_param=cfg["svdd"]["center_param"],
            svdd_radius_param=cfg["svdd"]["radius_param"],
            svdd_dropout=cfg["svdd"]["dropout"],
            svdd_use_batchnorm=cfg["svdd"]["use_batch_norm"]
        ).to(self.device)

        # loss, weight param
        self.recon_type = cfg["ae"].get("recon_type", "mae")
        self.ae_reduction = cfg["ae"].get("reduction", "mean")
        self.simsiam_lamda = cfg["ae"].get("simsiam_lamda", 1.0)
        self.svdd_lamda = cfg["ae"].get("svdd_lamda", 1.0)

        # criterion
        self.recon_criterion = ReconLoss(loss_type=self.recon_type, reduction=self.ae_reduction)
        self.simsiam_criterion = SimSiamLoss().to(self.device)
        self.svdd_criterion = DeepSVDDLoss(
            nu=cfg["svdd"].get("nu", 0.1),
            reduction=cfg["svdd"].get("reduction", "mean")
        ).to(self.device)

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

    def init_center(self, data_loader, eps=1e-5):
        """
        Initialize the center of the actual data distribution before training
        Args:
            data_loader: DataLoader containing only normal data (or entire dataset)
            eps (float): Constant for correcting values that are too close to zero
        """
        print("[DeepSVDD] Initializing center with mean of dataset ...")
        self.model.eval()

        # center vector
        latent_dim = self.model.svdd.latent_dim
        c = torch.zeros(latent_dim, device=self.device)
        n_samples = 0

        pbar = tqdm(enumerate(data_loader), total=len(data_loader), desc=f"Epoch [Init/{self.epochs}] Center", leave=True)
        with torch.no_grad():
            for batch_idx, data in pbar:
                (x_s, _, _), (x_t, _, _) = data  # x_s, x_t: (B, C, T)
                x_s = x_s.to(self.device)
                x_t = x_t.to(self.device)

                # forward
                (e_s, e_t, z_s, p_s, z_t, p_t, feat_s, dist_s, feat_t, dist_t, x_s_recon, x_t_recon) = self.model(x_s, x_t)
                c += torch.sum(feat_s, dim=0)
                n_samples += feat_s.size(0)

        c /= n_samples

        eps = eps
        mask = torch.abs(c) < eps
        c[mask] = 0.0
        self.model.svdd.center.data = c
        print(f"[DeepSVDD] center initialized. (norm={c.norm():.4f}, value={c[:5]})")

    def train_epoch(self, train_loader, epoch: int):
        do_train = (epoch > 0)
        if do_train:
            self.model.train()
        else:
            self.model.eval()
        total_loss = 0.0
        recons_loss = 0.0
        simsiam_loss = 0.0
        deep_svdd_loss = 0.0
        deep_svdd_loss_s = 0.0
        deep_svdd_loss_t = 0.0
        recons_loss_s = 0.0
        recons_loss_t = 0.0

        sum_dist_s = 0.0
        sum_dist_t = 0.0
        count_samples = 0
        #last_outputs = None

        pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch [{epoch}/{self.epochs}] Train", leave=False)
        for batch_idx, data in pbar:
            (x_s, y_s, _), (x_t, y_t, _) = data  # x_s, x_t: (B, C, T)
            x_s = x_s.to(self.device)
            y_s = y_s.to(self.device)
            x_t = x_t.to(self.device)
            y_t = y_t.to(self.device)
            
            self.optimizer.zero_grad()

            # forward
            (e_s, e_t, z_s, p_s, z_t, p_t, feat_s, feat_t, dist_s, dist_t, x_s_recon, x_t_recon) = self.model(x_s, x_t)
            
            # dist_s, dist_t: L2^2 distance (B,)
            sum_dist_s += dist_s.detach().sum().item()
            sum_dist_t += dist_t.detach().sum().item()
            count_samples += dist_s.size(0)

            # loss
            sim_loss = self.simsiam_criterion(p_s, z_t, p_t, z_s)
            svdd_loss_s = self.svdd_criterion(feat_s, self.model.svdd.center, self.model.svdd.radius)
            svdd_loss_t = self.svdd_criterion(feat_t, self.model.svdd.center, self.model.svdd.radius)
            svdd_loss = 0.5 * (svdd_loss_s + svdd_loss_t)
            recon_loss_s = self.recon_criterion(x_s_recon, x_s)  # (pred, target)
            recon_loss_t = self.recon_criterion(x_t_recon, x_t)
            recon_loss = 0.5 * (recon_loss_s + recon_loss_t)

            loss = recon_loss + self.simsiam_lamda * sim_loss + self.svdd_lamda * svdd_loss

            # backprop
            if do_train:
                loss.backward()
                self.optimizer.step()

            # stats
            total_loss += loss.item()
            recons_loss += recon_loss.item()
            simsiam_loss += sim_loss.item()
            deep_svdd_loss += svdd_loss.item()
            deep_svdd_loss_s += svdd_loss_s.item()
            deep_svdd_loss_t += svdd_loss_t.item()
            recons_loss_s += recon_loss_s.item()
            recons_loss_t += recon_loss_t.item()
            
            # mlflow log: global step
            if self.mlflow_logger is not None and batch_idx % 40 == 0:
                global_step = epoch * len(train_loader) + batch_idx
                self.mlflow_logger.log_metrics({"train_loss_step": loss.item(),"train_recon_step": recon_loss.item(), 
                                                "train_simsiam_step": sim_loss.item(), "train_svdd_step": svdd_loss.item(), }, step=global_step)
                
            # tqdm
            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "recon": f"{recon_loss.item():.4f}",
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
        avg_recon_loss = recons_loss / num_batches
        avg_recon_loss_s = recons_loss_s / num_batches
        avg_recon_loss_t = recons_loss_t / num_batches

        # calc dist_s, dist_t avg
        avg_dist_s = sum_dist_s / count_samples if count_samples > 0 else 0.0
        avg_dist_t = sum_dist_t / count_samples if count_samples > 0 else 0.0

        center_out = self.model.svdd.center
        radius_out = self.model.svdd.radius

        print(f"[Train] [Epoch {epoch}/{self.epochs}] "
              f"Avg: {avg_loss:.4f} | Recon: {avg_recon_loss:.4f} | "
              f"SimSiam: {avg_sim_loss:.4f} | SVDD: {avg_svdd_loss:.4f} | "
              f"SVDD_S: {avg_svdd_loss_s:.4f} | SVDD_T: {avg_svdd_loss_t:.4f} | "
              f"Dist_S: {avg_dist_s:.4f} | Dist_T: {avg_dist_t:.4f} | "
              f"Radius: {radius_out.item():.4f}")

        if self.mlflow_logger is not None:
            self.mlflow_logger.log_metrics({
                "train_loss": avg_loss,
                "train_recon_loss": avg_recon_loss,
                "train_recon_loss_s": avg_recon_loss_s,
                "train_recon_loss_t": avg_recon_loss_t,
                "train_simsiam_loss": avg_sim_loss,
                "train_svdd_loss": avg_svdd_loss,
                "train_svdd_loss_s": avg_svdd_loss_s,
                "train_svdd_loss_t": avg_svdd_loss_t,
                "train_mean_dist_s": avg_dist_s,
                "train_mean_dist_t": avg_dist_t,
                "train_radius": radius_out.item()
            }, step=epoch)
            
        return (avg_loss, avg_recon_loss, avg_sim_loss, avg_svdd_loss, avg_svdd_loss_s, avg_svdd_loss_t, avg_dist_s, avg_dist_t)

    def eval_epoch(self, eval_loader, epoch: int):
        self.model.eval()
        total_loss = 0.0
        recons_loss = 0.0
        simsiam_loss = 0.0
        deep_svdd_loss = 0.0
        deep_svdd_loss_s = 0.0
        deep_svdd_loss_t = 0.0
        recons_loss_s = 0.0
        recons_loss_t = 0.0

        sum_dist_s = 0.0
        sum_dist_t = 0.0
        count_samples = 0
        #last_outputs = None

        pbar = tqdm(enumerate(eval_loader), total=len(eval_loader), desc=f"Epoch [{epoch}/{self.epochs}] Eval", leave=False)
        with torch.no_grad():
            for batch_idx, data in pbar:
                (x_s, y_s, _), (x_t, y_t, _) = data
                x_s = x_s.to(self.device)
                y_s = y_s.to(self.device)
                x_t = x_t.to(self.device)
                y_t = y_t.to(self.device)

                # forward
                (e_s, e_t, z_s, p_s, z_t, p_t, feat_s, feat_t, dist_s, dist_t, x_s_recon, x_t_recon) = self.model(x_s, x_t)

                # dist_s, dist_t: L2^2 distance (B,)
                sum_dist_s += dist_s.detach().sum().item()
                sum_dist_t += dist_t.detach().sum().item()
                count_samples += dist_s.size(0)
                
                # loss
                sim_loss = self.simsiam_criterion(p_s, z_t, p_t, z_s)
                svdd_loss_s = self.svdd_criterion(feat_s, self.model.svdd.center, self.model.svdd.radius)
                svdd_loss_t = self.svdd_criterion(feat_t, self.model.svdd.center, self.model.svdd.radius)
                svdd_loss = 0.5 * (svdd_loss_s + svdd_loss_t)
                recon_loss_s = self.recon_criterion(x_s_recon, x_s)  # (pred, target)
                recon_loss_t = self.recon_criterion(x_t_recon, x_t)
                recon_loss = 0.5 * (recon_loss_s + recon_loss_t)

                loss = recon_loss + self.simsiam_lamda * sim_loss + self.svdd_lamda * svdd_loss

                # stats
                total_loss += loss.item()
                recons_loss += recon_loss.item()
                simsiam_loss += sim_loss.item()
                deep_svdd_loss += svdd_loss.item()
                deep_svdd_loss_s += svdd_loss_s.item()
                deep_svdd_loss_t += svdd_loss_t.item()
                recons_loss_s += recon_loss_s.item()
                recons_loss_t += recon_loss_t.item()

                # mlflow log: global step
                if self.mlflow_logger is not None and batch_idx % 2 == 0:
                    global_step = epoch * len(eval_loader) + batch_idx
                    self.mlflow_logger.log_metrics({"eval_loss_step": loss.item(), "eval_recon_step": recon_loss.item(), 
                                                    "eval_simsiam_step": sim_loss.item(), "eval_svdd_step": svdd_loss.item(), }, step=global_step)
                
                # tqdm
                pbar.set_postfix({
                    "loss": f"{loss.item():.4f}",
                    "recon": f"{recon_loss.item():.4f}",
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
        avg_recon_loss = recons_loss / num_batches
        avg_recon_loss_s = recons_loss_s / num_batches
        avg_recon_loss_t = recons_loss_t / num_batches

        # calc dist_s, dist_t avg
        avg_dist_s = sum_dist_s / count_samples if count_samples > 0 else 0.0
        avg_dist_t = sum_dist_t / count_samples if count_samples > 0 else 0.0

        center_out = self.model.svdd.center
        radius_out = self.model.svdd.radius

        print(f"[Eval]  [Epoch {epoch}/{self.epochs}] "
              f"Avg: {avg_loss:.4f} | Recon: {avg_recon_loss:.4f} | "
              f"SimSiam: {avg_sim_loss:.4f} | SVDD: {avg_svdd_loss:.4f} | "
              f"SVDD_S: {avg_svdd_loss_s:.4f} | SVDD_T: {avg_svdd_loss_t:.4f} | "
              f"Dist_S: {avg_dist_s:.4f} | Dist_T: {avg_dist_t:.4f} | "
              f"Radius: {radius_out.item():.4f}")
        
        if self.mlflow_logger is not None:
            self.mlflow_logger.log_metrics({
                "eval_loss": avg_loss,
                "eval_recon_loss": avg_recon_loss,
                "eval_recon_loss_s": avg_recon_loss_s,
                "eval_recon_loss_t": avg_recon_loss_t,
                "eval_simsiam_loss": avg_sim_loss,
                "eval_svdd_loss": avg_svdd_loss,
                "eval_svdd_loss_s": avg_svdd_loss_s,
                "eval_svdd_loss_t": avg_svdd_loss_t,
                "eval_mean_dist_s": avg_dist_s,
                "eval_mean_dist_t": avg_dist_t
            }, step=epoch)

        return (avg_loss, avg_recon_loss, avg_sim_loss, avg_svdd_loss, avg_svdd_loss_s, avg_svdd_loss_t, avg_dist_s, avg_dist_t)

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
        
        last_saved_epoch = None

        if self.model.svdd.center_param:
            self.init_center(train_loader, eps=1e-5)

        for epoch in range(self.epochs + 1):
            if not self.model.svdd.center_param:
                print(f"Epoch {epoch}: Initializing center")
                self.init_center(train_loader, eps=1e-5)

            train_loss_tuple = self.train_epoch(train_loader, epoch)  # train
            (train_avg, train_recon, train_sim, train_svdd, train_svdd_s, train_svdd_t, train_dist_s, train_dist_t) = train_loss_tuple

            eval_loss_tuple = None
            if eval_loader is not None and epoch % self.log_every == 0:
                eval_loss_tuple = self.eval_epoch(eval_loader, epoch)  # eval
                (eval_avg, eval_recon, eval_sim, eval_svdd, eval_svdd_s, eval_svdd_t, eval_dist_s, eval_dist_t) = eval_loss_tuple

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
        # MLflow Registry final model and return run_id, last_saved_epoch
        if self.mlflow_logger is not None:
            self.mlflow_logger.log_metrics({"last_epoch": last_saved_epoch})
            """
            final_ckpt_path = self.model_utils.get_file_path(self.model_utils.get_file_name(last_saved_epoch))
            self.mlflow_logger.register_model(model_path=final_ckpt_path, model_name=self.run_name)
            """
            run_id = self.mlflow_logger.run_id if self.mlflow_logger is not None else None
            run_id_path = self.model_utils.get_file_path(f"{self.run_name}.json")
            self.mlflow_logger.save_run_id(run_id_path)

            self.mlflow_logger.end_run()