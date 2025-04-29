from __future__ import annotations

import io
import logging
from typing import Iterable, TypeAlias

from lightning import pytorch as pl
import torch
from torch import Tensor, nn, optim

from chemprop.data import (
    BatchMolGraph,
    MolAtomBondTrainingBatch,
    MulticomponentTrainingBatch,
    TrainingBatch,
)
from chemprop.nn import Aggregation, ChempropMetric, MixedMessagePassing, Predictor
from chemprop.nn.transforms import ScaleTransform
from chemprop.schedulers import build_NoamLike_LRSched
from chemprop.utils.registry import Factory

logger = logging.getLogger(__name__)

BatchType: TypeAlias = TrainingBatch | MolAtomBondTrainingBatch | MulticomponentTrainingBatch


class MolAtomBondMPNN(pl.LightningModule):
    def __init__(
        self,
        message_passing: MixedMessagePassing,
        agg: Aggregation,
        mol_predictor: Predictor,
        atom_predictor: Predictor,
        bond_predictor: Predictor,
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
    def output_dim(self) -> list[int]:
        return [
            self.predictors[0].output_dim,
            self.predictors[1].output_dim,
            self.predictors[2].output_dim,
        ]

    @property
    def n_tasks(self) -> list[int]:
        return [self.predictors[0].n_tasks, self.predictors[1].n_tasks, self.predictors[2].n_tasks]

    @property
    def n_targets(self) -> list[int]:
        return [
            self.predictors[0].n_targets,
            self.predictors[1].n_targets,
            self.predictors[2].n_targets,
        ]

    @property
    def criterion(self) -> list[ChempropMetric]:
        return [
            self.predictors[0].criterion,
            self.predictors[1].criterion,
            self.predictors[2].criterion,
        ]

    def fingerprint(
        self,
        bmg: BatchMolGraph,
        V_d: Tensor | None = None,
        E_d: Tensor | None = None,
        X_d: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
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
    ) -> list[Tensor]:
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
    ) -> list[Tensor]:
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
    ) -> list[Tensor]:
        """Return the predictions of the input batch

        Parameters
        ----------
        batch : MixedTrainingBatch
            the input batch

        Returns
        -------
        Tensor
            a tensor of varying shape depending on the task type:

            * regression/binary classification: ``n x (t * s)``, where ``n`` is the number of input
              molecules/reactions, ``t`` is the number of tasks, and ``s`` is the number of targets
              per task. The final dimension is flattened, so that the targets for each task are
              grouped. I.e., the first ``t`` elements are the first target for each task, the second
              ``t`` elements the second target, etc.

            * multiclass classification: ``n x t x c``, where ``c`` is the number of classes
        """
        bmg, X_vd, X_ed, X_d, *_ = batch

        predss = self(bmg, X_vd, X_ed, X_d)
        predss[2] = (predss[2][::2] + predss[2][1::2]) / 2
        return predss

    def configure_optimizers(self):
        opt = optim.Adam(self.parameters(), self.init_lr)
        if self.trainer.train_dataloader is None:
            # Loading `train_dataloader` to estimate number of training batches.
            # Using this line of code can pypass the issue of using `num_training_batches` as described [here](https://github.com/Lightning-AI/pytorch-lightning/issues/16060).
            self.trainer.estimated_stepping_batches
        steps_per_epoch = self.trainer.num_training_batches
        warmup_steps = self.warmup_epochs * steps_per_epoch
        if self.trainer.max_epochs == -1:
            logger.warn(
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
