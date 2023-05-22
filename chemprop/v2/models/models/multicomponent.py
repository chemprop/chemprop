from abc import ABC, abstractmethod
from typing import Iterable, Optional, Sequence, Union
import warnings

from lightning import pytorch as pl
import torch
from torch import Tensor, nn, optim

from chemprop.v2.data.dataloader import TrainingBatch
from chemprop.v2.featurizers.molgraph import BatchMolGraph
from chemprop.v2.models.models.molecular import MolecularMPNN
from chemprop.v2.models.modules import MolecularMessagePassingBlock, Aggregation, OutputTransform
from chemprop.v2.models.loss import LossFunction
from chemprop.v2.models.metrics import Metric
from chemprop.v2.models.modules.agg import Aggregation
from chemprop.v2.models.modules.message_passing.molecule import MolecularInput
from chemprop.v2.models.schedulers import NoamLR


class MulticomponentMPNN(MolecularMPNN):
    def __init__(
        self,
        blocks: Sequence[MolecularMessagePassingBlock],
        n_components: int,
        agg: Aggregation,
        ffn: nn.Sequential,
        transform: OutputTransform,
        loss_fn: LossFunction,
        metrics: Iterable[Metric],
        shared: bool = False,
        task_weights: Tensor | None = None,
        warmup_epochs: int = 2,
        num_lrs: int = 1,
        init_lr: float = 1e-4,
        max_lr: float = 1e-3,
        final_lr: float = 1e-4,
    ):
        super().__init__()

        if len(blocks) == 0:
            raise ValueError("arg 'blocks' was empty!")

        if shared and len(blocks) > 1:
            warnings.warn(
                "More than 1 block was supplied but 'shared' was True! Using only the 0th block..."
            )
        elif len(blocks) != n_components and not shared:
            raise ValueError(
                "arg 'n_components' must be equal to `len(blocks)` if 'shared' is False! "
                f"got: {n_components} and {len(blocks)}, respectively."
            )
        
        self.n_components = n_components
        self.shared = shared

        if self.shared:
            self.components = nn.ModuleList([blocks[0] for _ in range(self.n_components)])
        else:
            self.components = nn.ModuleList(blocks)

        self.agg = agg
        self.ffn = ffn
        self.transform = transform
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
        return self.ffn[-1].out_features // self.transform.n_targets
    

    def fingerprint(
        self, bmgs: Iterable[BatchMolGraph], V_ds: Iterable[Tensor], V_f: Tensor | None = None
    ) -> Tensor:
        """Calculate the learned fingerprint for the input molecules"""
        Hs = []
        for block, bmg, V_d in zip(self.blocks, bmgs, V_ds):
            H_v = block(bmg, V_d)
            H = self.agg(H_v[1:], [n_a for _, n_a in bmg.a_scope])
            Hs.append(H)
        H = torch.cat(Hs, 1)

        return H if V_f is None else torch.cat((H, V_f), 1)

    def encoding(
        self, bmgs: Iterable[BatchMolGraph], V_ds: Iterable[Tensor], V_f: Tensor | None = None
    ) -> Tensor:
        """Calculate the encoding (i.e., final hidden representation) for the input molecules"""
        return self.ffn[:-1](self.fingerprint(bmgs, V_ds, V_f))

    def forward(
        self, bmgs: Iterable[BatchMolGraph], V_ds: Iterable[Tensor], V_f: Tensor | None = None
    ) -> Tensor:
        """Generate predictions for the input molecules/reactions"""
        return self.transform(self.ffn(self.fingerprint(bmgs, V_ds, V_f)))

    def training_step(self, batch, batch_idx):
        bmgs, V_ds, X_f, targets, weights, lt_targets, gt_targets = batch

        mask = targets.isfinite()
        targets = targets.nan_to_num(nan=0.0)

        preds = self(bmgs, V_ds, X_f)

        l = self.criterion(
            preds,
            targets,
            mask,
            weights,
            self.task_weights,
            lt_targets=lt_targets,
            gt_targets=gt_targets
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
        
    def validation_step(self, batch, batch_idx: int = 0) -> tuple[list[Tensor], int]:
        preds, *_ = self.predict_step(batch, batch_idx)
        *_, targets, _, lt_targets, gt_targets = batch

        losses = self.evaluate_preds(preds, targets, lt_targets, gt_targets)
        metric2loss = {f"val/{m.alias}": l for m, l in zip(self.metrics, losses)}

        self.log_dict(metric2loss, on_epoch=True, batch_size=len(targets))
        self.log("val_loss", losses[0], on_epoch=True, batch_size=len(targets), prog_bar=True)

    def test_step(self, batch: TrainingBatch, batch_idx: int = 0):
        preds, *_ = self.predict_step(batch, batch_idx)
        *_, targets, _, lt_targets, gt_targets = batch

        losses = self.evaluate_preds(preds, targets, lt_targets, gt_targets)
        metric2loss = {f"test/{m.alias}": l for m, l in zip(self.metrics, losses)}
        self.log_dict(metric2loss, on_epoch=True, batch_size=len(targets), prog_bar=True)

    def predict_step(
        self, batch, batch_idx: int, dataloader_idx: int = 0
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
        bmgs, V_ds, X_f, *_ = batch

        return self(bmgs, V_ds, X_f)

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
