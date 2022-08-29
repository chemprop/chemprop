from typing import Optional
import pytorch_lightning as pl

import torch
from torch import Tensor
from torch.optim import Adam

from chemprop.models.v2.models import MPNN
from chemprop.models.v2.ptl.loss import LossFunction
from chemprop.nn_utils import NoamLR

class PtlMPNN(pl.LightningModule):
    def __init__(
        self,
        mpnn: MPNN,
        criterion: LossFunction,
        metric: LossFunction,
        target_weights: Optional[Tensor] = None,
        warmup_epochs: int = 2,
        num_lrs: int = 1,
        init_lr: float = 1e-4,
        max_lr: float = 1e-3,
        final_lr: float = 1e-4
    ):
        self.save_hyperparameters()

        self.mpnn = mpnn
        self.criterion = criterion
        self.metric = metric
        if target_weights is not None:
            self.target_weights = target_weights
        else:
            self.target_weights = torch.ones(1, self.mpnn.n_targets)

        self.warmup_epochs = warmup_epochs
        self.num_lrs = num_lrs
        self.init_lr = init_lr
        self.max_lr = max_lr
        self.final_lr = final_lr

    def training_step(self, batch: tuple, batch_idx):
        Xs, Y, data_weights, lt_targets, gt_targets = batch

        mask = ~torch.isnan(Y)
        Y = torch.nan_to_num(Y, nan=0.0)
        target_weights = torch.ones(Y.shape[1]).unsqueeze(0)

        Y_pred = self.mpnn(*Xs)
        # if args.dataset_type == 'multiclass':
        #     targets = targets.long()
        #     loss = (
        #         torch.cat(
        #             [
        #                 loss_func(preds[:, target_index, :], targets[:, target_index]).unsqueeze(1)
        #                 for target_index in range(preds.size(1))
        #             ],
        #             dim=1
        #         )
        #         * class_weights * mask
        #     )

        L = self.criterion(
            Y_pred,
            Y,
            weights=data_weights,
            mask=mask,
            lt_targets=lt_targets,
            gt_targets=gt_targets
        )
        L = L * target_weights * mask

        return L.sum() / mask.sum()
    
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