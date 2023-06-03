from abc import ABC, abstractmethod
from itertools import chain
from typing import Iterable, Optional, Union

import pytorch_lightning as pl
import torch
from torch import Tensor, nn, optim

from chemprop.v2.data.dataloader import TrainingBatch
from chemprop.v2.models.modules import MessagePassingBlock, MolecularInput
from chemprop.v2.models.loss import LossFunction, build_loss
from chemprop.v2.models.metrics import Metric
from chemprop.v2.models.schedulers import NoamLR
from chemprop.v2.models.utils import get_activation_function


class MPNN(ABC, pl.LightningModule):
    """An `MPNN` is comprised of a `MessagePassingBlock` and an FFN top-model. The former
    calculates learned encodings from an input molecule/reaction graph, and the latter takes these
    encodings as input to calculate a final prediction. The full model is trained end-to-end.

    An `MPNN` takes a input a molecular graph and outputs a tensor of shape `b x t * s`, where `b`
    the size of the batch (i.e., number of molecules in the graph,) `t` is the number of tasks to
    predict, and `s` is the number of targets to predict per task.

    NOTE: the number of targets `s` is *not* related to the number of classes to predict.  It is
    used as a multiplier for the output dimension of the MPNN when the predictions correspond to a
    parameterized distribution, e.g., MVE regression, for which `s` is 2. Typically, this is just 1.
    """

    def __init__(
        self,
        mpn_block: MessagePassingBlock,
        n_tasks: int,
        ffn_hidden_dim: int = 300,
        ffn_num_layers: int = 1,
        dropout: float = 0.0,
        activation: str = "relu",
        criterion: Optional[Union[str, LossFunction]] = None,
        metrics: Optional[Iterable[Union[str, Metric]]] = None,
        task_weights: Optional[Tensor] = None,
        warmup_epochs: int = 2,
        num_lrs: int = 1,
        init_lr: float = 1e-4,
        max_lr: float = 1e-3,
        final_lr: float = 1e-4,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["mpn_block"])

        self.mpn_block = mpn_block
        self.ffn = self.build_ffn(
            mpn_block.output_dim,
            n_tasks * self.n_targets,
            ffn_hidden_dim,
            ffn_num_layers,
            dropout,
            activation,
        )

        self.n_tasks = n_tasks
        self.criterion = criterion
        self.metrics = metrics

        if task_weights is None:
            task_weights = torch.ones(self.n_tasks)
        self.task_weights = nn.Parameter(task_weights.unsqueeze(0), requires_grad=False)

        self.warmup_epochs = warmup_epochs
        self.num_lrs = num_lrs
        self.init_lr = init_lr
        self.max_lr = max_lr
        self.final_lr = final_lr

    @classmethod
    @property
    @abstractmethod
    def _DATASET_TYPE(self) -> str:
        """the dataset type of this MPNN"""

    @classmethod
    @property
    @abstractmethod
    def _DEFAULT_CRITERION(self) -> str:
        """the default criterion with which to train this MPNN"""

    @classmethod
    @property
    @abstractmethod
    def _DEFAULT_METRIC(self) -> str:
        """the default metric with which to evaluate this MPNN"""

    @property
    def criterion(self) -> LossFunction:
        return self.__criterion

    @criterion.setter
    def criterion(self, criterion: Optional[Union[str, LossFunction]]):
        """Set the criterion with which to train this MPNN using its string alias or initialized
        `LossFunction` object"""

        if criterion is None:
            criterion = build_loss(self._DATASET_TYPE, self._DEFAULT_CRITERION)
        elif isinstance(criterion, str):
            criterion = build_loss(self._DATASET_TYPE, criterion)

        self.__criterion = criterion

    @property
    def metrics(self) -> Iterable[Metric]:
        return self.__metrics

    @metrics.setter
    def metrics(self, metrics: Optional[Iterable[Union[str, Metric]]]):
        """Set the evaluation metrics for this MPNN using their string aliases or initialized
        `Metric` objects"""
        if metrics is None:
            metrics = [self._DEFAULT_METRIC]

        metrics_ = []
        for m in metrics:
            try:
                metrics_.append(Metric.registry[m]() if isinstance(m, str) else m)
            except KeyError:
                raise ValueError(
                    f"Invalid metric! got: {m}. expected one of: {Metric.registry.keys()}"
                )

        self.__metrics = metrics_

    @property
    def n_targets(self) -> int:
        return 1

    @staticmethod
    def build_ffn(
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 300,
        n_layers: int = 1,
        dropout: float = 0.0,
        activation: str = "relu",
    ) -> nn.Sequential:
        dropout = nn.Dropout(dropout)
        activation = get_activation_function(activation)
        layers = [hidden_dim] * n_layers

        layers = [input_dim, *layers, output_dim]
        ffn = list(
            chain(
                *(
                    (dropout, nn.Linear(d1, d2), activation)
                    for d1, d2 in zip(layers[:-1], layers[1:])
                )
            )
        )

        return nn.Sequential(*ffn[:-1])

    def fingerprint(
        self, inputs: Union[MolecularInput, Iterable[MolecularInput]], X_f: Optional[Tensor] = None
    ) -> Tensor:
        """Calculate the learned fingerprint for the input molecules/reactions"""
        H = self.mpn_block(*inputs)
        if X_f is not None:
            H = torch.cat((H, X_f), 1)

        return H

    def encoding(
        self, inputs: Union[MolecularInput, Iterable[MolecularInput]], X_f: Optional[Tensor] = None
    ) -> Tensor:
        """Calculate the encoding ("hidden representation") for the input molecules/reactions"""
        return self.ffn[:-1](self.fingerprint(inputs, X_f=X_f))

    def forward(
        self, inputs: Union[MolecularInput, Iterable[MolecularInput]], X_f: Optional[Tensor] = None
    ) -> Tensor:
        """Generate predictions for the input molecules/reactions.

        NOTE: the type signature of `input` matches the underlying `encoder.forward()`
        """
        return self.ffn(self.fingerprint(inputs, X_f=X_f))

    def training_step(self, batch: TrainingBatch, batch_idx):
        bmg, X_vd, features, targets, weights, lt_targets, gt_targets = batch

        mask = targets.isfinite()
        targets = targets.nan_to_num(nan=0.0)

        preds = self((bmg, X_vd), X_f=features)

        l = self.criterion(
            preds, targets, mask, weights=weights, lt_targets=lt_targets, gt_targets=gt_targets
        )

        self.log("train/loss", l, prog_bar=True)

        return l

    def validation_step(self, batch: TrainingBatch, batch_idx: int = 0) -> tuple[list[Tensor], int]:
        *_, targets, _, lt_targets, gt_targets = batch

        preds, _ = self.predict_step(batch, batch_idx)

        mask = targets.isfinite()
        targets = targets.nan_to_num(nan=0.0)

        losses = [
            metric(preds, targets, mask, lt_targets=lt_targets, gt_targets=gt_targets)
            for metric in self.metrics
        ]
        metric2loss = {f"val/{m.alias}": l for m, l in zip(self.metrics, losses)}
        self.log_dict(metric2loss, on_epoch=True, batch_size=len(targets), prog_bar=True)

    def test_step(self, batch: TrainingBatch, batch_idx: int = 0):
        *_, targets, _, lt_targets, gt_targets = batch

        preds, _ = self.predict_step(batch, batch_idx)

        mask = targets.isfinite()
        targets = targets.nan_to_num(nan=0.0)

        losses = [
            metric(preds, targets, mask, lt_targets=lt_targets, gt_targets=gt_targets)
            for metric in self.metrics
        ]
        metric2loss = {f"{m.alias}": l for m, l in zip(self.metrics, losses)}

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
        bmg, X_vd, features, *_ = batch

        return (self((bmg, X_vd), X_f=features), None)  # TODO: include uncertainty in this

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
