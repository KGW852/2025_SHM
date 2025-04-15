# trainers/convergent_sim_trainer.py

import torch
from tqdm import tqdm

from models.convergent_sim import SimEncoder
from models.criterions.simsiam_loss import SimSiamLoss
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

        # optimizer, scheduler, early stopper
        self.optimizer = Optimizer(self.cfg).get_optimizer(self.encoder.parameters())
        self.scheduler = Scheduler(self.cfg).get_scheduler(self.optimizer)
        self.early_stopper = EarlyStopper(self.cfg).get_early_stopper()

        # utils: model manage, mlflow
        self.model_utils = ModelUtils(self.cfg)
        self.mlflow_logger = mlflow_logger

        # params
        self.epochs = cfg["epochs"]
        self.log_every = cfg.get("log_every", 1)
        self.save_every = cfg.get("save_every", 1)
        self.eval_n_batch = cfg["eval_n_batch"]

        # run_name: model_name
        self.run_name = self.model_utils.get_model_name()

    def train_epoch(self, train_loader, epoch: int):
        do_train = (epoch > 0)
        if do_train:
            self.encoder.train()
        else:
            self.encoder.eval()
        total_loss = 0.0

        pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch [{epoch}/{self.epochs}] Train", leave=False)
        for batch_idx, data in pbar:
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
            if do_train:
                loss.backward()
                self.optimizer.step()

            # stats
            total_loss += loss.item()
            pbar.set_postfix({"loss": loss.item()})
        
        # scheduler
        if do_train and (self.scheduler is not None):
            self.scheduler.step()
            
        avg_loss = total_loss / len(train_loader)
        print(f"[Train] [Epoch {epoch}/{self.epochs}] "
              f"SimSiam: {avg_loss:.4f}")

        return avg_loss

    def eval_epoch(self, eval_loader, epoch: int):
        self.encoder.eval()
        total_loss = 0.0

        pbar = tqdm(enumerate(eval_loader), total=len(eval_loader), desc=f"Epoch [{epoch}/{self.epochs}] Eval", leave=False)
        with torch.no_grad():
            for batch_idx, data in pbar:
                (x_s, y_s), (x_t, y_t) = data
                x_s = x_s.to(self.device)
                y_s = y_s.to(self.device)
                x_t = x_t.to(self.device)
                y_t = y_t.to(self.device)

                e_s, e_t, z_s, p_s, z_t, p_t = self.encoder(x_s, x_t)
                sim_loss = self.simsiam_criterion(p_s, z_t, p_t, z_s)
                loss = sim_loss

                total_loss += loss.item()
                pbar.set_postfix({"loss": loss.item()})

        avg_loss = total_loss / len(eval_loader)
        print(f"[Eval]  [Epoch {epoch}/{self.epochs}] "
              f"SimSiam: {avg_loss:.4f}")

        return avg_loss

    def get_embeddings(self, data_loader):
        """
        Iterate through the entire data_loader, extracting (e_s, e_t) embeddings from the encoder.
        Collect the extracted embeddings into a list (or concatenate into a tensor) and return them.
        """
        self.encoder.eval()
        src_f_enc = []
        tgt_f_enc = []
        src_f_proj = []
        tgt_f_proj = []
        src_class_lbl = []
        tgt_class_lbl = []

        # The smaller value between the total length of the current data_loader and eval_n_batch
        if not self.eval_n_batch or self.eval_n_batch <= 0:
            max_batches = len(data_loader)
        else:
            max_batches = min(len(data_loader), self.eval_n_batch)

        pbar = tqdm(enumerate(data_loader), total=max_batches, desc=f"Embedding [{data_loader}]", leave=False)
        with torch.no_grad():
            for batch_idx, data in pbar:
                (x_s, y_s), (x_t, y_t) = data
                x_s = x_s.to(self.device)
                class_y_s = y_s[:,0].to(self.device)  # (B,)
                x_t = x_t.to(self.device)
                y_t = y_t.to(self.device)
                class_y_t = y_t[:,0].to(self.device)  # (B,)

                # forward
                e_s, e_t, z_s, p_s, z_t, p_t = self.encoder(x_s, x_t)

                # stack
                src_f_enc.append(e_s.detach().cpu())
                tgt_f_enc.append(e_t.detach().cpu())
                src_f_proj.append(z_s.detach().cpu())
                tgt_f_proj.append(z_t.detach().cpu())
                src_class_lbl.append(class_y_s.detach().cpu())
                tgt_class_lbl.append(class_y_t.detach().cpu())

                if (batch_idx + 1) >= max_batches:
                    break

        # combine the embeddings of all batches into a single Tensor
        src_f_enc = torch.cat(src_f_enc, dim=0)
        tgt_f_enc = torch.cat(tgt_f_enc, dim=0)
        src_f_proj = torch.cat(src_f_proj, dim=0)
        tgt_f_proj = torch.cat(tgt_f_proj, dim=0)
        src_class_lbl = torch.cat(src_class_lbl, dim=0)
        tgt_class_lbl = torch.cat(tgt_class_lbl, dim=0)

        return src_f_enc, tgt_f_enc, src_f_proj, tgt_f_proj, src_class_lbl, tgt_class_lbl

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

        for epoch in range(self.epochs + 1):
            train_loss = self.train_epoch(train_loader, epoch)  # train

            eval_loss = None
            if eval_loader is not None and epoch % self.log_every == 0:
                eval_loss = self.eval_epoch(eval_loader, epoch)  # eval

                # calc umap, NN distace
                src_f_enc, tgt_f_enc, src_f_proj, tgt_f_proj, src_class_lbl, tgt_class_lbl = self.get_embeddings(eval_loader)  # extract embedding
                plot_latent_alignment(cfg=self.cfg, mlflow_logger=self.mlflow_logger, 
                                      src_embed=src_f_enc, tgt_embed=tgt_f_enc, src_lbl=src_class_lbl, tgt_lbl=tgt_class_lbl, 
                                      epoch=epoch, f_class="encoder")
                plot_latent_alignment(cfg=self.cfg, mlflow_logger=self.mlflow_logger, 
                                      src_embed=src_f_proj, tgt_embed=tgt_f_proj, src_lbl=src_class_lbl, tgt_lbl=tgt_class_lbl, 
                                      epoch=epoch, f_class="projector")

            # mlflow metrics log
            if self.mlflow_logger is not None:
                metrics = {"train_loss": train_loss}
                if eval_loss is not None:
                    metrics["eval_loss"] = eval_loss
                self.mlflow_logger.log_metrics(metrics, step=epoch)

            if eval_loss is not None and self.early_stopper is not None:  # early stopping check(use val loss)
                if self.early_stopper.step(eval_loss):
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