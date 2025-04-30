from __future__ import annotations

import io
import logging
from typing import Iterable

from lightning import pytorch as pl
import torch
from torch import Tensor, nn, optim

from chemprop.data import BatchMolGraph, MolAtomBondTrainingBatch
from chemprop.nn import Aggregation, ChempropMetric, MABMessagePassing, Predictor
from chemprop.nn.transforms import ScaleTransform
from chemprop.schedulers import build_NoamLike_LRSched
from chemprop.utils.registry import Factory

logger = logging.getLogger(__name__)


class MolAtomBondMPNN(pl.LightningModule):
    r"""An :class:`MolAtomBondMPNN` is a sequence of message passing layers, an aggregation routine,
    and up to three predictor routines for molecule-level, atom-level, and bond-level predictions.

    The first two modules calculate learned graph, node, and edge embeddings from an input molecule
    graph, and the final modules take these learned fingerprints as input to calculate a
    final prediction. I.e., the following operation:

    .. math::
        \mathtt{MPNN}(\mathcal{G}) =
            \mathtt{predictor}(\mathtt{agg}(\mathtt{message\_passing}(\mathcal{G})))

    The full model is trained end-to-end.

    Parameters
    ----------
    message_passing : MABMessagePassing
        the message passing block to use to calculate learned fingerprints
    agg : Aggregation
        the aggregation operation to use during molecule-level prediction
    mol_predictor : Predictor
        the function to use to calculate the final molecule-level prediction
    atom_predictor : Predictor
        the function to use to calculate the final atom-level prediction
    bond_predictor : Predictor
        the function to use to calculate the final bond-level prediction
    batch_norm : bool, default=False
        if `True`, apply batch normalization to the learned fingerprints before passing them to the
        predictors
    metrics : Iterable[Metric] | None, default=None
        the metrics to use to evaluate the model during training and evaluation
    warmup_epochs : int, default=2
        the number of epochs to use for the learning rate warmup
    init_lr : int, default=1e-4
        the initial learning rate
    max_lr : float, default=1e-3
        the maximum learning rate
    final_lr : float, default=1e-4
        the final learning rate

    Raises
    ------
    ValueError
        if the output dimension of the message passing block does not match the input dimension of
        the predictor functions
    ValueError
        if none of the predictor functions are provided
    ValueError
        if `mol_predictor` is provided but `agg` is not
    """

    def __init__(
        self,
        message_passing: MABMessagePassing,
        agg: Aggregation | None = None,
        mol_predictor: Predictor | None = None,
        atom_predictor: Predictor | None = None,
        bond_predictor: Predictor | None = None,
        batch_norm: bool = False,
        metrics: Iterable[ChempropMetric] | None = None,
        warmup_epochs: int = 2,
        init_lr: float = 1e-4,
        max_lr: float = 1e-3,
        final_lr: float = 1e-4,
        X_d_transform: ScaleTransform | None = None,
    ):
        super().__init__()
        # manually add X_d_transform to hparams to suppress lightning's warning about double saving
        # its state_dict values.
        self.save_hyperparameters(
            ignore=[
                "X_d_transform",
                "message_passing",
                "agg",
                "mol_predictor",
                "atom_predictor",
                "bond_predictor",
            ]
        )
        self.hparams["X_d_transform"] = X_d_transform
        self.hparams.update(
            {
                "message_passing": message_passing.hparams,
                "agg": agg.hparams,
                "mol_predictor": mol_predictor.hparams,
                "atom_predictor": atom_predictor.hparams,
                "bond_predictor": bond_predictor.hparams,
            }
        )

        self.message_passing = message_passing
        self.agg = agg
        self.bn = nn.BatchNorm1d(self.message_passing.output_dim) if batch_norm else nn.Identity()
        self.predictors = nn.ModuleList([mol_predictor, atom_predictor, bond_predictor])

        self.X_d_transform = X_d_transform if X_d_transform is not None else nn.Identity()

        self.metrics = nn.ModuleList(
            [
                nn.ModuleList([*metrics, self.criterion[i].clone()])
                if metrics
                else nn.ModuleList(
                    [self.predictors[i]._T_default_metric(), self.criterion[i].clone()]
                )
                for i in range(3)
            ]
        )

        self.warmup_epochs = warmup_epochs
        self.init_lr = init_lr
        self.max_lr = max_lr
        self.final_lr = final_lr

    @property
    def output_dimss(self) -> tuple[int | None, int | None, int | None]:
        return tuple(
            predictor.output_dim if predictor is not None else None for predictor in self.predictors
        )

    @property
    def n_taskss(self) -> tuple[int | None, int | None, int | None]:
        return tuple(
            predictor.n_tasks if predictor is not None else None for predictor in self.predictors
        )

    @property
    def n_targetss(self) -> tuple[int | None, int | None, int | None]:
        return tuple(
            predictor.n_targets if predictor is not None else None for predictor in self.predictors
        )

    @property
    def criterions(
        self,
    ) -> tuple[ChempropMetric | None, ChempropMetric | None, ChempropMetric | None]:
        return tuple(
            predictor.criterion if predictor is not None else None for predictor in self.predictors
        )

    def fingerprint(
        self,
        bmg: BatchMolGraph,
        V_d: Tensor | None = None,
        E_d: Tensor | None = None,
        X_d: Tensor | None = None,
    ) -> tuple[Tensor | None, Tensor | None, Tensor | None]:
        """the learned fingerprints for the input molecules"""
        H_v, H_b = self.message_passing(bmg, V_d, E_d)
        H_g = self.agg(H_v, bmg.batch)
        H_g = self.bn[0](H_g)

        H_g = H_g if X_d is None else torch.cat((H_g, self.X_d_transform(X_d)), 1)
        H_b = torch.cat([H_b, H_b[bmg.rev_edge_index]], 1)
        return [H_g, H_v, H_b]

    def encoding(
        self,
        bmg: BatchMolGraph,
        V_d: Tensor | None = None,
        E_d: Tensor | None = None,
        X_d: Tensor | None = None,
        i: int = -1,
    ) -> tuple[Tensor | None, Tensor | None, Tensor | None]:
        """Calculate the :attr:`i`-th hidden representation"""
        H = self.fingerprint(bmg, V_d, E_d, X_d)
        return [
            self.predictors[0].encode(H[0], i),
            self.predictors[1].encode(H[1], i),
            self.predictors[2].encode(H[2], i),
        ]

    def forward(
        self,
        bmg: BatchMolGraph,
        V_d: Tensor | None = None,
        E_d: Tensor | None = None,
        X_d: Tensor | None = None,
    ) -> tuple[Tensor | None, Tensor | None, Tensor | None]:
        """Generate predictions for the input molecules/reactions"""
        H = self.fingerprint(bmg, V_d, E_d, X_d)
        return [self.predictors[0](H[0]), self.predictors[1](H[1]), self.predictors[2](H[2])]

    def training_step(self, batch: MolAtomBondTrainingBatch, batch_idx):
        total_l = 0

        bmg, V_d, E_d, X_d, targets, weights, lt_masks, gt_masks = batch
        Z = self.fingerprint(bmg, V_d, E_d, X_d)
        for index in range(len(targets)):
            if targets[index] is None:
                continue

            mask = targets[index].isfinite()
            targets[index] = targets[index].nan_to_num(nan=0.0)
            preds = self.predictors[index].train_step(Z[index])
            if index == 2:
                preds = (preds[::2] + preds[1::2]) / 2
            l = self.criterion[index](
                preds, targets[index], mask, weights[index], lt_masks[index], gt_masks[index]
            )
            total_l += l

        return total_l

    def on_validation_model_eval(self) -> None:
        self.eval()
        self.predictors[0].output_transform.train()
        self.predictors[1].output_transform.train()
        self.predictors[2].output_transform.train()

    def validation_step(self, batch: MolAtomBondTrainingBatch, batch_idx: int = 0):
        self._evaluate_batch(batch, "val")

        bmg, V_d, E_d, X_d, targets, weights, lt_masks, gt_masks = batch
        Z = self.fingerprint(bmg, V_d, E_d, X_d)
        agg_metric = 0
        for index in range(len(targets)):
            if targets[index] is None:
                continue

            mask = targets[index].isfinite()
            targets[index] = targets[index].nan_to_num(nan=0.0)
            preds = self.predictors[index].train_step(Z[index])
            if index == 2:
                preds = (preds[::2] + preds[1::2]) / 2

            self.metrics[index][-1](
                preds, targets[index], mask, weights[index], lt_masks[index], gt_masks[index]
            )
            agg_metric += self.metrics[index][-1].compute()
            self.metrics[index][-1].reset()

        self.log("val_loss", agg_metric, batch_size=len(batch[0]), prog_bar=True)

    def test_step(self, batch: MolAtomBondTrainingBatch, batch_idx: int = 0):
        self._evaluate_batch(batch, "test")

    def _evaluate_batch(self, batch: MolAtomBondTrainingBatch, label: str) -> None:
        bmg, V_d, E_d, X_d, targets, weights, lt_masks, gt_masks = batch
        for index in range(len(targets)):
            if targets[index] is None:
                continue

            if index == 0:
                label = "mol_" + label
            elif index == 1:
                label = "atom_" + label
            else:
                label = "bond_" + label

            mask = targets[index].isfinite()
            targets[index] = targets[index].nan_to_num(nan=0.0)
            preds = self(bmg, V_d, E_d, X_d)
            preds = preds[index]
            if index == 2:
                preds = (preds[::2] + preds[1::2]) / 2
            weights[index] = torch.ones_like(weights[index])
            if self.predictors[index].n_targets > 1:
                preds = preds[..., 0]

            for m in self.metrics[index][:-1]:
                m.update(
                    preds, targets[index], mask, weights[index], lt_masks[index], gt_masks[index]
                )
                self.log(f"{label}/{m.alias}", m, batch_size=len(batch[0]))

    def predict_step(
        self, batch: MolAtomBondTrainingBatch, batch_idx: int, dataloader_idx: int = 0
    ) -> tuple[Tensor | None, Tensor | None, Tensor | None]:
        bmg, V_d, E_d, X_d, *_ = batch

        return self(bmg, V_d, E_d, X_d)

    def configure_optimizers(self):
        opt = optim.Adam(self.parameters(), self.init_lr)
        if self.trainer.train_dataloader is None:
            # Loading `train_dataloader` to estimate number of training batches.
            # Using this line of code can pypass the issue of using `num_training_batches` as described [here](https://github.com/Lightning-AI/pytorch-lightning/issues/16060).
            self.trainer.estimated_stepping_batches
        steps_per_epoch = self.trainer.num_training_batches
        warmup_steps = self.warmup_epochs * steps_per_epoch
        if self.trainer.max_epochs == -1:
            logger.warning(
                "For infinite training, the number of cooldown epochs in learning rate scheduler is set to 100 times the number of warmup epochs."
            )
            cooldown_steps = 100 * warmup_steps
        else:
            cooldown_epochs = self.trainer.max_epochs - self.warmup_epochs
            cooldown_steps = cooldown_epochs * steps_per_epoch

        lr_sched = build_NoamLike_LRSched(
            opt, warmup_steps, cooldown_steps, self.init_lr, self.max_lr, self.final_lr
        )

        lr_sched_config = {"scheduler": lr_sched, "interval": "step"}

        return {"optimizer": opt, "lr_scheduler": lr_sched_config}

    @classmethod
    def _load(cls, path, map_location, **submodules):
        d = torch.load(path, map_location, weights_only=False)

        try:
            hparams = d["hyper_parameters"]
            state_dict = d["state_dict"]
        except KeyError:
            raise KeyError(f"Could not find hyper parameters and/or state dict in {path}.")

        submodules |= {
            key: hparams[key].pop("cls")(**hparams[key])
            for key in (
                "message_passing",
                "agg",
                "mol_predictor",
                "atom_predictor",
                "bond_predictor",
            )
            if key not in submodules
        }

        if not hasattr(submodules["mol_predictor"].criterion, "_defaults"):
            submodules["mol_predictor"].criterion = submodules["mol_predictor"].criterion.__class__(
                task_weights=submodules["mol_predictor"].criterion.task_weights
            )

        return submodules, state_dict, hparams

    @classmethod
    def _add_metric_task_weights_to_state_dict(cls, state_dict, hparams):
        return state_dict

    @classmethod
    def load_from_checkpoint(
        cls, checkpoint_path, map_location=None, hparams_file=None, strict=True, **kwargs
    ) -> MolAtomBondMPNN:
        submodules = {
            k: v for k, v in kwargs.items() if k in ["message_passing", "agg", "predictor"]
        }
        submodules, state_dict, hparams = cls._load(checkpoint_path, map_location, **submodules)
        kwargs.update(submodules)

        state_dict = cls._add_metric_task_weights_to_state_dict(state_dict, hparams)
        d = torch.load(checkpoint_path, map_location, weights_only=False)
        d["state_dict"] = state_dict
        buffer = io.BytesIO()
        torch.save(d, buffer)
        buffer.seek(0)

        return super().load_from_checkpoint(buffer, map_location, hparams_file, strict, **kwargs)

    @classmethod
    def load_from_file(
        cls, model_path, map_location=None, strict=True, **submodules
    ) -> MolAtomBondMPNN:
        submodules, state_dict, hparams = cls._load(model_path, map_location, **submodules)
        hparams.update(submodules)
        model = cls(**hparams)
        model.load_state_dict(state_dict, strict=strict)

        return model
