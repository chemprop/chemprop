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
        X_d: Tensor | None = None,
    ) -> Tensor:
        H_vs: list[Tensor] = self.message_passing(bmgs, V_ds)
        Hs = [self.agg(H_v, bmg.batch) for H_v, bmg in zip(H_vs, bmgs)]
        H = torch.cat(Hs, 1)
        H = self.bn(H)

        return H if X_d is None else torch.cat((H, X_d), 1)

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

    @classmethod
    def load_from_file(cls, model_path, map_location=None, strict=True) -> MPNN:
        d = torch.load(model_path, map_location=map_location)

        try:
            hparams = d["hyper_parameters"]
            state_dict = d["state_dict"]
        except KeyError:
            raise KeyError(f"Could not find hyper parameters and/or state dict in {model_path}. ")

        for key in ["message_passing", "agg", "predictor"]:
            hparam_kwargs = hparams[key]
            if key == "message_passing":
                hparam_kwargs["blocks"] = [
                    block_hparams.pop("cls")(**block_hparams)
                    for block_hparams in hparam_kwargs["blocks"]
                ]
            hparam_cls = hparam_kwargs.pop("cls")
            hparams[key] = hparam_cls(**hparam_kwargs)

        model = cls(**hparams)
        model.load_state_dict(state_dict, strict=strict)

        return model
