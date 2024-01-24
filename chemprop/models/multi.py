from typing import Iterable

import torch
from torch import Tensor

from chemprop.data import BatchMolGraph
from chemprop.nn import MulticomponentMessagePassing, Aggregation, Predictor
from chemprop.models.model import MPNN
from chemprop.nn.metrics import Metric


class MulticomponentMPNN(MPNN):
    def __init__(
        self,
        message_passing: MulticomponentMessagePassing,
        agg: Aggregation,
        predictor: Predictor,
        batch_norm: bool = True,
        metrics: Iterable[Metric] | None = None,
        w_t: Tensor | None = None,
        warmup_epochs: int = 2,
        init_lr: float = 1e-4,
        max_lr: float = 1e-3,
        final_lr: float = 1e-4,
    ):
        super().__init__(
            message_passing,
            agg,
            predictor,
            batch_norm,
            metrics,
            w_t,
            warmup_epochs,
            init_lr,
            max_lr,
            final_lr,
        )
        self.message_passing: MulticomponentMessagePassing

    def fingerprint(
        self,
        bmgs: Iterable[BatchMolGraph],
        V_ds: Iterable[Tensor | None],
        X_f: Tensor | None = None,
    ) -> Tensor:
        H_vs: list[Tensor] = self.message_passing(bmgs, V_ds)
        Hs = [self.agg(H_v, bmg.batch) for H_v, bmg in zip(H_vs, bmgs)]
        H = torch.cat(Hs, 1)
        H = self.bn(H)

        return H if X_f is None else torch.cat((H, X_f), 1)

    @classmethod
    def load_from_checkpoint(
        cls, checkpoint_path, map_location=None, hparams_file=None, strict=True, **kwargs
    ) -> MPNN:
        hparams = torch.load(checkpoint_path)["hyper_parameters"]

        mp_hparams = hparams["message_passing"]
        mp_hparams["blocks"] = [
            block_hparams.pop("cls")(**block_hparams) for block_hparams in mp_hparams["blocks"]
        ]
        message_passing = mp_hparams.pop("cls")(**mp_hparams)
        kwargs["message_passing"] = message_passing

        return super().load_from_checkpoint(
            checkpoint_path, map_location, hparams_file, strict, **kwargs
        )
