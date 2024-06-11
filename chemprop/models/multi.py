from typing import Iterable

import torch
from torch import Tensor, nn

from chemprop.data import BatchMolGraph
from chemprop.models.model import MPNN
from chemprop.nn import Aggregation, MulticomponentMessagePassing, Predictor
from chemprop.nn.metrics import Metric
from chemprop.nn.transforms import GraphTransform, ScaleTransform, UnscaleTransform


class MulticomponentMPNN(MPNN):
    def __init__(
        self,
        message_passing: MulticomponentMessagePassing,
        agg: Aggregation,
        predictor: Predictor,
        batch_norm: bool = True,
        metrics: Iterable[Metric] | None = None,
        warmup_epochs: int = 2,
        init_lr: float = 1e-4,
        max_lr: float = 1e-3,
        final_lr: float = 1e-4,
        X_d_transform: ScaleTransform | None = None,
    ):
        super().__init__(
            message_passing,
            agg,
            predictor,
            batch_norm,
            metrics,
            warmup_epochs,
            init_lr,
            max_lr,
            final_lr,
            X_d_transform,
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

        return H if X_d is None else torch.cat((H, self.X_d_transform(X_d)), 1)

    @classmethod
    def load_submodules(cls, path, map_location, **kwargs):
        d = torch.load(path, map_location)

        try:
            hparams = d["hyper_parameters"]
            state_dict = d["state_dict"]
        except KeyError:
            raise KeyError(f"Could not find hyper parameters and/or state dict in {path}.")

        i = 0
        while True:
            if f"message_passing.blocks.{i}.V_d_transform.mean" in state_dict:
                dummy_tensor = (
                    torch.zeros_like(state_dict[f"message_passing.blocks.{i}.V_d_transform.mean"])
                    .squeeze(0)
                    .numpy()
                )
                hparams["message_passing"]["blocks"][i]["V_d_transform"] = ScaleTransform(
                    mean=dummy_tensor, scale=dummy_tensor
                )

            if f"message_passing.blocks.{i}.graph_transform.E_transform.mean" in state_dict:
                dummy_tensor = (
                    torch.zeros_like(
                        state_dict[f"message_passing.blocks.{i}.graph_transform.E_transform.mean"]
                    )
                    .squeeze(0)
                    .numpy()
                )
                E_f_transform = ScaleTransform(mean=dummy_tensor, scale=dummy_tensor)
            else:
                E_f_transform = nn.Identity()

            if f"message_passing.blocks.{i}.graph_transform.V_transform.mean" in state_dict:
                dummy_tensor = (
                    torch.zeros_like(
                        state_dict[f"message_passing.blocks.{i}.graph_transform.V_transform.mean"]
                    )
                    .squeeze(0)
                    .numpy()
                )
                V_f_transform = ScaleTransform(mean=dummy_tensor, scale=dummy_tensor)
            else:
                V_f_transform = nn.Identity()

            if isinstance(E_f_transform, nn.Identity) and isinstance(V_f_transform, nn.Identity):
                pass
            else:
                hparams["message_passing"]["blocks"][i]["graph_transform"] = GraphTransform(
                    V_f_transform, E_f_transform
                )

            try:
                i += 1
                state_dict[f"message_passing.blocks.{i}.W_i.weight"]
            except KeyError:
                break

            if hparams["message_passing"]["shared"]:
                break

        if "predictor.output_transform.mean" in state_dict:
            dummy_tensor = (
                torch.zeros_like(state_dict["predictor.output_transform.mean"]).squeeze(0).numpy()
            )
            hparams["predictor"]["output_transform"] = UnscaleTransform(
                mean=dummy_tensor, scale=dummy_tensor
            )

        if "X_d_transform.mean" in state_dict and "X_d_transform" not in kwargs:
            dummy_tensor = torch.zeros_like(state_dict["X_d_transform.mean"]).squeeze(0).numpy()
            kwargs["X_d_transform"] = ScaleTransform(mean=dummy_tensor, scale=dummy_tensor)

        hparams["message_passing"]["blocks"] = [
            block_hparams.pop("cls")(**block_hparams)
            for block_hparams in hparams["message_passing"]["blocks"]
        ]
        kwargs |= {
            key: hparams[key].pop("cls")(**hparams[key])
            for key in ("message_passing", "agg", "predictor")
            if key not in kwargs
        }
        return kwargs, state_dict
