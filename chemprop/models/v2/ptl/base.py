from abc import ABC, abstractmethod
from typing import Iterable, Optional, Union

import pytorch_lightning as pl
import torch
from torch import Tensor, optim
from torchmetrics import Metric

from chemprop.data.v2.dataloader import TrainingBatch
from chemprop.models.v2.models import MPNN
from chemprop.models.v2.models.loss import BCELoss, LossFunction, MSELoss, MVELoss
from chemprop.models.v2.models.metrics import (
    BoundedMSEMetric, BoundedRMSEMetric, MSEMetric, Metric, RMSEMetric
)
from chemprop.nn_utils import NoamLR


class LitMPNNBase(ABC, pl.LightningModule):
    def __init__(
        self,
        mpnn: MPNN,
        criterion: Union[str, LossFunction],
        metrics: Iterable[Union[str, Metric]],
        task_weights: Optional[Tensor] = None,
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
        self.metrics = metrics

        if task_weights is not None:
            self.task_weights = task_weights
        else:
            self.task_weights = torch.ones(self.mpnn.n_tasks)

        self.warmup_epochs = warmup_epochs
        self.num_lrs = num_lrs
        self.init_lr = init_lr
        self.max_lr = max_lr
        self.final_lr = final_lr

        if not isinstance(self.criterion, self.valid_mpnns):
            raise ValueError
        if not isinstance(self.criterion, self.valid_criteria):
            raise ValueError(f"Invalid criterion! Expected one of: {self.valid_criteria}.")
        for metric in self.metrics:
            if not isinstance(metric, self.valid_metrics):
                raise ValueError(f"Invalid metric! Expected one of: {self.valid_criteria}.")

    @property
    def valid_mpnns(self) -> tuple[MPNN, ...]:
        return (MPNN,)

    @property
    def valid_criteria(self) -> tuple[LossFunction, ...]:
        return (MSELoss, MVELoss, BCELoss)
    
    @property
    def valid_metrics(self) -> tuple[Metric, ...]:
        return (MSEMetric, RMSEMetric, BoundedMSEMetric, BoundedRMSEMetric)

    def calc_loss(self, preds, targets, mask, data_weights, lt_targets, gt_targets) -> Tensor:
        loss = self.criterion(
            preds, targets, mask=mask, lt_targets=lt_targets, gt_targets=gt_targets
        )

        return loss * data_weights * mask

    def training_step(self, batch: TrainingBatch, batch_idx):
        bmg, X_vd, features, targets, data_weights, lt_targets, gt_targets = batch

        mask = targets.isfinite()
        targets = targets.nan_to_num(nan=0.0)

        preds = self.mpnn(bmg, X_vd, X_f=features)

        L = self.calc_loss(preds, targets, mask, data_weights, lt_targets, gt_targets)
        L = L * self.task_weights

        return L.sum() / mask.sum()
    
    def validation_step(self, batch: TrainingBatch, batch_idx) -> Tensor:
        bmg, X_vd, features, targets, _, lt_targets, gt_targets = batch

        mask = targets.isfinite()
        preds = self.mpnn(bmg, X_vd, X_f=features)



    def predict_step(self, batch: TrainingBatch, batch_idx: int, dataloader_idx: int = 0):
        bmg, X_vd, features, *_ = batch

        return self.mpnn(bmg, X_vd, X_f=features)

    
