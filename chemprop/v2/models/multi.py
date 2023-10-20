from typing import Iterable

import torch
from torch import Tensor, nn

from chemprop.v2.data.dataloader import MulticomponentTrainingBatch
from chemprop.v2.featurizers.molgraph import BatchMolGraph
from chemprop.v2.models.model import MPNN
from chemprop.v2.models.modules import MulticomponentMessagePassing, Aggregation, Readout
from chemprop.v2.models.loss import LossFunction
from chemprop.v2.models.metrics import Metric


class MulticomponentMPNN(MPNN):
    def __init__(
        self,
        message_passing: MulticomponentMessagePassing,
        agg: Aggregation,
        readout: Readout,
        batch_norm: bool = True,
        metrics: Iterable[Metric] | None = None,
        w_t: Tensor | None = None,
        warmup_epochs: int = 2,
        init_lr: float = 1e-4,
        max_lr: float = 1e-3,
        final_lr: float = 1e-4,
    ):
        super().__init__(
            message_passing, # MPNN doesn't take MulticomponentMessagePassing?
            agg,
            readout,
            batch_norm,
            metrics,
            w_t,
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
        return self.readout[:-1](self.fingerprint(bmgs, V_ds, X_f))

    def forward(
        self, bmgs: Iterable[BatchMolGraph], V_ds: Iterable[Tensor], X_f: Tensor | None = None
    ) -> Tensor:
        """Generate predictions for the input molecules/reactions"""
        return self.readout(self.fingerprint(bmgs, V_ds, X_f))

    def training_step(self, batch: MulticomponentTrainingBatch, batch_idx):
        return super().training_step(batch, batch_idx)

    def validation_step(self, batch: MulticomponentTrainingBatch, batch_idx: int = 0):
        return super().validation_step(batch, batch_idx)

    def test_step(self, batch: MulticomponentTrainingBatch, batch_idx: int = 0):
        return super().test_step(batch, batch_idx)
