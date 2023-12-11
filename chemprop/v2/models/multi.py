from typing import Iterable

import torch
from torch import Tensor, nn

from chemprop.v2.data.collate import MulticomponentTrainingBatch, BatchMolGraph
from chemprop.v2.nn import (
    MulticomponentMessagePassing,
    Aggregation,
    # OutputTransform,
    LossFunction,
    Metric,
)
from chemprop.v2.models.model import MPNN

OutputTransform = None


class MulticomponentMPNN(MPNN):
    def __init__(
        self,
        message_passing: MulticomponentMessagePassing,
        agg: Aggregation,
        ffn: nn.Sequential,
        transform: OutputTransform,
        loss_fn: LossFunction,
        metrics: Iterable[Metric],
        task_weights: Tensor | None = None,
        warmup_epochs: int = 2,
        init_lr: float = 1e-4,
        max_lr: float = 1e-3,
        final_lr: float = 1e-4,
    ):
        super().__init__(
            message_passing,
            agg,
            ffn,
            transform,
            loss_fn,
            metrics,
            task_weights,
            warmup_epochs,
            init_lr,
            max_lr,
            final_lr,
        )

    def fingerprint(
        self, bmgs: Iterable[BatchMolGraph], V_ds: Iterable[Tensor], X_f: Tensor | None = None
    ) -> Tensor:
        H_vs = self.message_passing(bmgs, V_ds)
        H = [self.agg(H_v[1:], bmg.a_scope) for H_v, bmg in zip(H_vs, bmgs)]

        return H if X_f is None else torch.cat((H, X_f), 1)

    def encoding(
        self, bmgs: Iterable[BatchMolGraph], V_ds: Iterable[Tensor], X_f: Tensor | None = None
    ) -> Tensor:
        """Calculate the encoding (i.e., final hidden representation) for the input molecules"""
        return self.ffn[:-1](self.fingerprint(bmgs, V_ds, X_f))

    def forward(
        self, bmgs: Iterable[BatchMolGraph], V_ds: Iterable[Tensor], X_f: Tensor | None = None
    ) -> Tensor:
        """Generate predictions for the input molecules/reactions"""
        return self.transform(self.ffn(self.fingerprint(bmgs, V_ds, X_f)))

    def training_step(self, batch: MulticomponentTrainingBatch, batch_idx):
        return super().training_step(batch, batch_idx)

    def validation_step(self, batch: MulticomponentTrainingBatch, batch_idx: int = 0):
        return super().validation_step(batch, batch_idx)

    def test_step(self, batch: MulticomponentTrainingBatch, batch_idx: int = 0):
        return super().test_step(batch, batch_idx)
