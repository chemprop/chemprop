from abc import ABC, abstractmethod
from typing import Optional
import pytorch_lightning as pl

import torch
from torch import Tensor
from torch.optim import Adam

from chemprop.models.v2.models import MPNN
from chemprop.models.v2.ptl import loss
from chemprop.nn_utils import NoamLR

class LitMPNNBase(ABC, pl.LightningModule):
    def __init__(
        self,
        mpnn: MPNN,
        criterion: loss.LossFunction,
        metric: loss.LossFunction,
        target_weights: Optional[Tensor] = None,
        warmup_epochs: int = 2,
        num_lrs: int = 1,
        init_lr: float = 1e-4,
        max_lr: float = 1e-3,
        final_lr: float = 1e-4
    ):
        super().__init__()
        self.save_hyperparameters()

        self.mpnn = mpnn
        self.criterion = criterion
        self.metric = metric

        if target_weights is not None:
            self.target_weights = target_weights
        else:
            self.target_weights = torch.ones(self.mpnn.n_targets)

        self.warmup_epochs = warmup_epochs
        self.num_lrs = num_lrs
        self.init_lr = init_lr
        self.max_lr = max_lr
        self.final_lr = final_lr

        if not isinstance(self.criterion, self.valid_criteria):
            raise ValueError(f"Invalid criterion! Expected one of: {self.valid_criteria}.")
        if not isinstance(self.metric, self.valid_criteria):
            raise ValueError(f"Invalid metric! Expected one of: {self.valid_criteria}.")

    @property
    @abstractmethod
    def valid_criteria(self) -> tuple[loss.LossFunction]:
        pass

    def calc_loss(self, preds, targets, data_weights, mask, lt_targets, gt_targets) -> Tensor:
        loss = self.criterion(preds, targets, lt_targets=lt_targets, gt_targets=gt_targets)

        return loss * data_weights * mask

    def training_step(self, batch: tuple, batch_idx):
        inputs, targets, data_weights, lt_targets, gt_targets = batch

        mask = ~torch.isnan(targets)
        targets = torch.nan_to_num(targets, nan=0.0)

        preds = self.mpnn(*inputs)

        loss = self.calc_loss(preds, targets, data_weights, mask, lt_targets, gt_targets)
        loss = loss * self.target_weights

        return loss.sum() / mask.sum()
    
    def configure_optimizers(self):
        opt = Adam(self.mpnn.parameters(), self.init_lr)

        lr_sched = NoamLR(
            optimizer=opt,
            warmup_epochs=[self.warmup_epochs],
            total_epochs=[self.trainer.max_epochs] * self.num_lrs,
            steps_per_epoch=self.num_training_steps,
            init_lr=[self.init_lr],
            max_lr=[self.max_lr],
            final_lr=[self.final_lr],
        )
        lr_sched_config = {
            "scheduler": lr_sched,
            "interval": "step" if isinstance(lr_sched, NoamLR) else "batch",
        }

        return {
            "optimizer": opt,
            "lr_scheduler": lr_sched_config
        }

    @property
    def num_training_steps(self) -> int:
        """Total training steps inferred from datamodule and devices."""
        if self.trainer.max_steps:
            return self.trainer.max_steps

        limit_batches = self.trainer.limit_train_batches
        batches = len(self.train_dataloader())

        if isinstance(limit_batches, int):
            batches = min(batches, limit_batches)
        else:
            batches = int(limit_batches * batches)

        num_devices = max(1, self.trainer.num_gpus, self.trainer.num_processes)
        if self.trainer.tpu_cores:
            num_devices = max(num_devices, self.trainer.tpu_cores)

        effective_accum = self.trainer.accumulate_grad_batches * num_devices
        return (batches // effective_accum) * self.trainer.max_epochs


class MCCClassicficationMPNN(LitMPNNBase):
    @property
    def valid_criteria(self) -> tuple[loss.LossFunction]:
        return (loss.ClassificationMCCLoss,)

    def calc_loss(self, preds, targets, data_weights, mask, lt_targets, gt_targets) -> Tensor:
        return self.criterion(preds, targets, mask, data_weights)


class MCCMulticlassMPNN(LitMPNNBase):
    @property
    def valid_criteria(self) -> tuple[loss.LossFunction]:
        return (loss.MulticlassMCCLoss,)
        
    def calc_loss(self, preds, targets, data_weights, mask, lt_targets, gt_targets) -> Tensor:
        targets = targets.long()
        target_losses = [
            self.criterion(preds[:, j], targets[:, j], data_weights, mask[:, j]).unsqueeze(0)
            for j in range(preds.shape[1])
        ]
        
        return torch.cat(target_losses)


class DirichletMulticlassMPNN(LitMPNNBase):
    @property
    def valid_criteria(self) -> tuple[loss.LossFunction]:
        return (loss.DirichletMulticlassLoss,)

    def calc_loss(self, preds, targets, data_weights, mask, lt_targets, gt_targets) -> Tensor:
        return self.criterion(preds, targets.long()) * data_weights * mask


class CrossEntropyMulticlassMPNN(LitMPNNBase):
    @property
    def valid_criteria(self) -> tuple[loss.LossFunction]:
        return (loss.CrossEntropyLoss,)

    def calc_loss(self, preds, targets, data_weights, mask, lt_targets, gt_targets) -> Tensor:
        targets = targets.long()
        target_losses = [
            self.criterion(preds[:, j], targets[:, j]).unsqueeze(1)
            for j in range(preds.shape[1])
        ]
        
        return torch.cat(target_losses, dim=1) * data_weights * mask
        

class LitSpectralMPNN(LitMPNNBase):
    @property
    def valid_criteria(self) -> tuple[loss.LossFunction]:
        return (loss.SIDSpectralLoss, loss.WassersteinSpectralLoss)

    def calc_loss(self, preds, targets, data_weights, mask, lt_targets, gt_targets) -> Tensor:
        return self.criterion(preds, targets, mask)