from abc import ABC, abstractmethod
from typing import Iterable, Optional, Union

from lightning import pytorch as pl
import torch
from torch import Tensor, nn, optim

from chemprop.v2.data.dataloader import TrainingBatch
from chemprop.v2.featurizers.molgraph import BatchMolGraph
from chemprop.v2.models.modules import MessagePassingBlockBase, Aggregation, OutputTransform
from chemprop.v2.models.loss import LossFunction, build_loss
from chemprop.v2.models.metrics import Metric, MetricFactory
from chemprop.v2.models.modules.agg import Aggregation, MeanAggregation
from chemprop.v2.models.modules.ffn import FFN
from chemprop.v2.models.schedulers import NoamLR


class MolecularMPNN(ABC, pl.LightningModule):
    """An `MPNN` is comprised of message passing layer, an aggregation routine, and an FFN
    top-model. The first two calculate learned encodings from an input molecule/reaction graph, and
    the latter takes these encodings as input to calculate a final prediction. The full model is
    trained end-to-end.

    An `MPNN` takes a input a molecular graph and outputs a tensor of shape `b x t * s`, where `b`
    the size of the batch (i.e., number of molecules in the graph,) `t` is the number of tasks to
    predict, and `s` is the number of targets to predict per task.

    NOTE: the number of targets `s` is *not* related to the number of classes to predict.  It is
    used as a multiplier for the output dimension of the MPNN when the predictions correspond to a
    parameterized distribution, e.g., MVE regression, for which `s` is 2. Typically, this is just 1.
    """

    def __init__(
        self,
        message_passing: MessagePassingBlockBase,
        agg: Aggregation | None,
        ffn: FFN,
        transform: OutputTransform | None,
        loss_fn: LossFunction,
        metrics: Iterable[Metric],
        task_weights: Tensor | None = None,
        warmup_epochs: int = 2,
        num_lrs: int = 1,
        init_lr: float = 1e-4,
        max_lr: float = 1e-3,
        final_lr: float = 1e-4,
    ):
        super().__init__()

        if message_passing.output_dim != ffn.input_dim:
            raise ValueError
        
        self.message_passing = message_passing
        self.agg = agg or MeanAggregation()
        self.ffn = ffn
        self.transform = transform or nn.Identity()
        self.criterion = loss_fn
        self.metrics = metrics

        if task_weights is None:
            task_weights = torch.ones(self.n_tasks)
        self.task_weights = nn.Parameter(task_weights.unsqueeze(0), requires_grad=False)

        self.warmup_epochs = warmup_epochs
        self.num_lrs = num_lrs
        self.init_lr = init_lr
        self.max_lr = max_lr
        self.final_lr = final_lr

    @property
    def n_tasks(self) -> int:
        return self.ffn.output_dim // self.transform.n_targets
    
    @property
    def metrics(self) -> list[Metric]:
        """The metrics this model will use to evaluate predictions during valiation and testing. By
        convention, the 0th metric will _also_ be logged as 'val_loss' in addition to its true
        alias"""
        return self.__metrics

    def fingerprint(
        self, bmg: BatchMolGraph, V_d: Tensor | None = None, X_f: Tensor | None = None
    ) -> Tensor:
        """Calculate the learned fingerprint for the input molecules"""
        H_v = self.message_passing(bmg, V_d)
        H = self.agg(H_v[1:], bmg.a_scope)

        return H if X_f is None else torch.cat((H, X_f), 1)

    def encoding(
        self, bmg: BatchMolGraph, V_d: Tensor | None = None, X_f: Tensor | None = None
    ) -> Tensor:
        """Calculate the encoding (i.e., final hidden representation) for the input molecules"""
        return self.ffn[:-1](self.fingerprint(bmg, V_d, X_f))

    def forward(
        self, bmg: BatchMolGraph, V_d: Tensor | None = None, X_f: Tensor | None = None
    ) -> Tensor:
        """Generate predictions for the input molecules/reactions"""
        return self.transform(self.ffn(self.fingerprint(bmg, V_d, X_f)))

    def training_step(self, batch: TrainingBatch, batch_idx):
        bmg, V_d, X_f, targets, weights, lt_mask, gt_mask = batch

        mask = targets.isfinite()
        targets = targets.nan_to_num(nan=0.0)

        preds = self(bmg, V_d, X_f)

        l = self.criterion(
            preds,
            targets,
            mask,
            weights,
            self.task_weights,
            lt_mask=lt_mask,
            gt_mask=gt_mask
        )

        self.log("train/loss", l, prog_bar=True)

        return l

    def evaluate_preds(self, preds, targets, lt_targets, gt_targets) -> list[Tensor]:
        mask = targets.isfinite()
        targets = targets.nan_to_num(nan=0.0)

        return [
            metric(preds, targets, mask, lt_targets=lt_targets, gt_targets=gt_targets)
            for metric in self.metrics
        ]
        
    def validation_step(self, batch: TrainingBatch, batch_idx: int = 0) -> tuple[list[Tensor], int]:
        *_, targets, _, lt_targets, gt_targets = batch
        preds, *_ = self.predict_step(batch, batch_idx)

        losses = self.evaluate_preds(preds, targets, lt_targets, gt_targets)
        metric2loss = {f"val/{m.alias}": l for m, l in zip(self.metrics, losses)}

        self.log_dict(metric2loss, on_epoch=True, batch_size=len(targets))
        self.log("val_loss", losses[0], on_epoch=True, batch_size=len(targets), prog_bar=True)

    def test_step(self, batch: TrainingBatch, batch_idx: int = 0):
        *_, targets, _, lt_targets, gt_targets = batch
        preds, *_ = self.predict_step(batch, batch_idx)

        losses = self.evaluate_preds(preds, targets, lt_targets, gt_targets)
        metric2loss = {f"test/{m.alias}": l for m, l in zip(self.metrics, losses)}
        self.log_dict(metric2loss, on_epoch=True, batch_size=len(targets), prog_bar=True)

    def predict_step(
        self, batch: TrainingBatch, batch_idx: int, dataloader_idx: int = 0
    ) -> tuple[Tensor, ...]:
        """Return the predictions of the input batch

        Parameters
        ----------
        batch : TrainingBatch
            the input batch

        Returns
        -------
        tuple[Tensor, ...]
            an n-tuple containing the predictions in the 0th index and uncertainty parameters for
            all remaining indices
        """
        bmg, X_vd, X_f, *_ = batch

        return self(bmg, X_vd, X_f)

    def configure_optimizers(self):
        opt = optim.Adam(self.parameters(), self.init_lr)

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

        return {"optimizer": opt, "lr_scheduler": lr_sched_config}

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
