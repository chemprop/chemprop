from __future__ import annotations

import io
import logging
from typing import Iterable

from lightning import pytorch as pl
import torch
from torch import Tensor, nn, optim

from chemprop.data import BatchMolAtomBondGraph, BatchMolGraph, MolAtomBondTrainingBatch
from chemprop.nn import Aggregation, ChempropMetric, ConstrainerFFN, MABMessagePassing, Predictor
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
    agg : Aggregation | None, default=None
        the aggregation operation to use during molecule-level prediction
    mol_predictor : Predictor | None, default=None
        the function to use to calculate the final molecule-level prediction
    atom_predictor : Predictor | None, default=None
        the function to use to calculate the final atom-level prediction
    bond_predictor : Predictor | None, default=None
        the function to use to calculate the final bond-level prediction
    atom_constrainer : ConstrainerFFN | None, default=None
        the constrainer to use to constrain the atom-level predictions
    bond_constrainer : ConstrainerFFN | None, default=None
        the constrainer to use to constrain the bond-level predictions
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
        atom_constrainer: ConstrainerFFN | None = None,
        bond_constrainer: ConstrainerFFN | None = None,
        batch_norm: bool = False,
        metrics: Iterable[ChempropMetric] | None = None,
        warmup_epochs: int = 2,
        init_lr: float = 1e-4,
        max_lr: float = 1e-3,
        final_lr: float = 1e-4,
        X_d_transform: ScaleTransform | None = None,
    ):
        if all([p is None for p in [mol_predictor, atom_predictor, bond_predictor]]):
            raise ValueError(
                "At least one of mol_predictor, atom_predictor, or bond_predictor must be provided."
            )
        if mol_predictor is not None and agg is None:
            raise ValueError("If mol_predictor is provided, agg must also be provided.")

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
                "atom_constrainer",
                "bond_constrainer",
            ]
        )
        self.hparams["X_d_transform"] = X_d_transform
        self.hparams.update(
            {
                "message_passing": message_passing.hparams,
                "agg": agg.hparams if agg is not None else None,
                "mol_predictor": mol_predictor.hparams if mol_predictor is not None else None,
                "atom_predictor": atom_predictor.hparams if atom_predictor is not None else None,
                "bond_predictor": bond_predictor.hparams if bond_predictor is not None else None,
                "atom_constrainer": atom_constrainer.hparams
                if atom_constrainer is not None
                else None,
                "bond_constrainer": bond_constrainer.hparams
                if bond_constrainer is not None
                else None,
            }
        )

        self.message_passing = message_passing
        self.agg = agg
        self.mol_predictor = mol_predictor
        self.atom_predictor = atom_predictor
        self.atom_constrainer = atom_constrainer

        if bond_predictor is not None:

            def wrapped_bond_fn(m, fn):
                def _wrapped_fn(*args, **kwargs):
                    preds = getattr(m, f"_{fn}")(*args, **kwargs)
                    preds = (preds[::2] + preds[1::2]) / 2
                    return preds

                return _wrapped_fn

            bond_predictor._forward = bond_predictor.forward
            bond_predictor._train_step = bond_predictor.train_step
            bond_predictor.forward = wrapped_bond_fn(bond_predictor, "forward")
            bond_predictor.train_step = wrapped_bond_fn(bond_predictor, "train_step")

            if bond_constrainer is not None:
                bond_constrainer.ffn._forward = bond_constrainer.ffn.forward
                bond_constrainer.ffn.forward = wrapped_bond_fn(bond_constrainer.ffn, "forward")

        self.bond_predictor = bond_predictor
        self.bond_constrainer = bond_constrainer
        self.predictors = [self.mol_predictor, self.atom_predictor, self.bond_predictor]

        fp_dims = [self.message_passing.output_dims[0]] * 2 + [self.message_passing.output_dims[1]]
        self.bns = nn.ModuleList(
            [
                nn.BatchNorm1d(dim) if batch_norm and predictor is not None else nn.Identity()
                for dim, predictor in zip(fp_dims, self.predictors)
            ]
        )

        self.X_d_transform = X_d_transform if X_d_transform is not None else nn.Identity()

        self.metricss = nn.ModuleList(
            [
                None
                if predictor is None
                else nn.ModuleList(
                    [metric.clone() for metric in metrics] + [predictor.criterion.clone()]
                )
                if metrics
                else nn.ModuleList([predictor._T_default_metric(), predictor.criterion.clone()])
                for predictor in self.predictors
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
        H_v, H_e = self.message_passing(bmg, V_d, E_d)
        H_g = self.agg(H_v, bmg.batch) if self.agg is not None else None

        H_g = self.bns[0](H_g) if H_g is not None else None
        H_v = self.bns[1](H_v) if H_v is not None else None
        H_e = self.bns[2](H_e) if H_e is not None else None

        H_g = (
            H_g
            if X_d is None
            else torch.cat((H_g, self.X_d_transform(X_d)), dim=1)
            if H_g is not None
            else None
        )
        H_e = torch.cat([H_e, H_e[bmg.rev_edge_index]], dim=1) if H_e is not None else None
        return H_g, H_v, H_e

    def encoding(
        self,
        bmg: BatchMolGraph,
        V_d: Tensor | None = None,
        E_d: Tensor | None = None,
        X_d: Tensor | None = None,
        i: int = -1,
    ) -> tuple[Tensor | None, Tensor | None, Tensor | None]:
        """Calculate the :attr:`i`-th hidden representation"""
        Hs = self.fingerprint(bmg, V_d, E_d, X_d)
        return tuple(
            predictor.encode(H, i) if predictor is not None else None
            for H, predictor in zip(Hs, self.predictors)
        )

    def forward(
        self,
        bmg: BatchMolAtomBondGraph,
        V_d: Tensor | None = None,
        E_d: Tensor | None = None,
        X_d: Tensor | None = None,
        constraints: tuple[Tensor | None, Tensor | None] | None = None,
    ) -> tuple[Tensor | None, Tensor | None, Tensor | None]:
        """Generate predictions for the input molecules/reactions"""
        fps = self.fingerprint(bmg, V_d, E_d, X_d)
        predss = [
            predictor(fp) if predictor is not None else None
            for fp, predictor in zip(fps, self.predictors)
        ]

        if constraints is not None:
            constrained_predss = []
            for constrainer, fp, preds, batch, constraint_tensor in zip(
                [None, self.atom_constrainer, self.bond_constrainer],
                fps,
                predss,
                [None, bmg.batch, bmg.bond_batch[::2]],
                [None] + constraints,
            ):
                if constrainer is None:
                    constrained_predss.append(preds)
                else:
                    if preds.ndim > 2:
                        preds[..., 0] = constrainer(fp, preds[..., 0], batch, constraint_tensor)
                    else:
                        preds = constrainer(fp, preds, batch, constraint_tensor)
                    constrained_predss.append(preds)
            predss = constrained_predss

        return predss

    def training_step(self, batch: MolAtomBondTrainingBatch, batch_idx):
        bmg, V_d, E_d, X_d, targetss, weightss, lt_masks, gt_masks, constraints = batch
        fps = self.fingerprint(bmg, V_d, E_d, X_d)
        predss = [
            predictor.train_step(fp) if predictor is not None else None
            for fp, predictor in zip(fps, self.predictors)
        ]

        if constraints is not None:
            constrained_predss = []
            for constrainer, fp, preds, batch, constraint_tensor in zip(
                [None, self.atom_constrainer, self.bond_constrainer],
                fps,
                predss,
                [None, bmg.batch, bmg.bond_batch[::2]],
                [None] + constraints,
            ):
                if constrainer is None:
                    constrained_predss.append(preds)
                else:
                    if preds.ndim > 2:
                        preds[..., 0] = constrainer(fp, preds[..., 0], batch, constraint_tensor)
                    else:
                        preds = constrainer(fp, preds, batch, constraint_tensor)
                    constrained_predss.append(preds)
            predss = constrained_predss

        total_l = 0
        for predictor, preds, targets, weights, lt_mask, gt_mask, kind in zip(
            self.predictors, predss, targetss, weightss, lt_masks, gt_masks, ["mol", "atom", "bond"]
        ):
            if predictor is None:
                continue

            mask = targets.isfinite()
            targets = targets.nan_to_num(nan=0.0)
            l = predictor.criterion(preds, targets, mask, weights, lt_mask, gt_mask)
            total_l += l
            self.log(
                f"{kind}_train_loss",
                predictor.criterion,
                batch_size=targets.shape[0],
                prog_bar=False,
                on_epoch=True,
            )

        n_datapoints = bmg.batch[-1].item() + 1
        self.log("train_loss", total_l, batch_size=n_datapoints, prog_bar=True, on_epoch=True)
        return total_l

    def on_validation_model_eval(self) -> None:
        self.eval()
        self.message_passing.V_d_transform.train()
        self.message_passing.E_d_transform.train()
        self.message_passing.graph_transform.train()
        self.X_d_transform.train()
        [
            predictor.output_transform.train()
            for predictor in self.predictors
            if predictor is not None
        ]

    def validation_step(self, batch: MolAtomBondTrainingBatch, batch_idx: int = 0):
        self._evaluate_batch(batch, "val")

        bmg, V_d, E_d, X_d, targetss, weightss, lt_masks, gt_masks, constraints = batch
        fps = self.fingerprint(bmg, V_d, E_d, X_d)
        predss = [
            predictor.train_step(fp) if predictor is not None else None
            for fp, predictor in zip(fps, self.predictors)
        ]

        if constraints is not None:
            constrained_predss = []
            for constrainer, fp, preds, batch, constraint_tensor in zip(
                [None, self.atom_constrainer, self.bond_constrainer],
                fps,
                predss,
                [None, bmg.batch, bmg.bond_batch[::2]],
                [None] + constraints,
            ):
                if constrainer is None:
                    constrained_predss.append(preds)
                else:
                    if preds.ndim > 2:
                        preds[..., 0] = constrainer(fp, preds[..., 0], batch, constraint_tensor)
                    else:
                        preds = constrainer(fp, preds, batch, constraint_tensor)
                    constrained_predss.append(preds)
            predss = constrained_predss

        total_vl = 0
        for predictor, preds, targets, weights, lt_mask, gt_mask, kind, metrics in zip(
            self.predictors,
            predss,
            targetss,
            weightss,
            lt_masks,
            gt_masks,
            ["mol", "atom", "bond"],
            self.metricss,
        ):
            if predictor is None:
                continue

            mask = targets.isfinite()
            targets = targets.nan_to_num(nan=0.0)
            vl = metrics[-1](preds, targets, mask, weights, lt_mask, gt_mask)
            total_vl += vl
            self.log(f"{kind}_val_loss", metrics[-1], batch_size=targets.shape[0], prog_bar=False)

        n_datapoints = bmg.batch[-1].item() + 1
        self.log("val_loss", total_vl, batch_size=n_datapoints, prog_bar=True)

    def test_step(self, batch: MolAtomBondTrainingBatch, batch_idx: int = 0):
        self._evaluate_batch(batch, "test")

    def _evaluate_batch(self, batch: MolAtomBondTrainingBatch, label: str) -> None:
        bmg, V_d, E_d, X_d, targetss, weightss, lt_masks, gt_masks, constraints = batch
        predss = self(bmg, V_d, E_d, X_d, constraints)

        for preds, predictor, targets, weights, lt_mask, gt_mask, kind, metrics in zip(
            predss,
            self.predictors,
            targetss,
            weightss,
            lt_masks,
            gt_masks,
            ["mol", "atom", "bond"],
            self.metricss,
        ):
            if predictor is None:
                continue

            mask = targets.isfinite()
            targets = targets.nan_to_num(nan=0.0)
            weights = torch.ones_like(weights)
            if predictor.n_targets > 1:
                preds = preds[..., 0]

            for m in metrics[:-1]:
                m.update(preds, targets, mask, weights, lt_mask, gt_mask)
                self.log(f"{kind}_{label}/{m.alias}", m, batch_size=targets.shape[0])

    def predict_step(
        self, batch: MolAtomBondTrainingBatch, batch_idx: int, dataloader_idx: int = 0
    ) -> tuple[Tensor | None, Tensor | None, Tensor | None]:
        bmg, V_d, E_d, X_d, *_, constraints = batch

        return self(bmg, V_d, E_d, X_d, constraints)

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

        if hparams["metrics"] is not None:
            hparams["metrics"] = [
                cls._rebuild_metric(metric)
                if not torch.cuda.is_available() and metric.device.type != "cpu"
                else metric
                for metric in hparams["metrics"]
            ]

        if (
            hparams["mol_predictor"] is not None
            and hparams["mol_predictor"]["criterion"] is not None
        ):
            metric = hparams["mol_predictor"]["criterion"]
            if not torch.cuda.is_available() and metric.device.type != "cpu":
                hparams["mol_predictor"]["criterion"] = cls._rebuild_metric(metric)

        if (
            hparams["atom_predictor"] is not None
            and hparams["atom_predictor"]["criterion"] is not None
        ):
            metric = hparams["atom_predictor"]["criterion"]
            if not torch.cuda.is_available() and metric.device.type != "cpu":
                hparams["atom_predictor"]["criterion"] = cls._rebuild_metric(metric)

        if (
            hparams["bond_predictor"] is not None
            and hparams["bond_predictor"]["criterion"] is not None
        ):
            metric = hparams["bond_predictor"]["criterion"]
            if not torch.cuda.is_available() and metric.device.type != "cpu":
                hparams["bond_predictor"]["criterion"] = cls._rebuild_metric(metric)

        submodules |= {
            key: hparams[key].pop("cls")(**hparams[key])
            for key in (
                "message_passing",
                "agg",
                "mol_predictor",
                "atom_predictor",
                "bond_predictor",
                "atom_constrainer",
                "bond_constrainer",
            )
            if key not in submodules and hparams[key] is not None
        }

        return submodules, state_dict, hparams

    @classmethod
    def _rebuild_metric(cls, metric):
        return Factory.build(metric.__class__, task_weights=metric.task_weights, **metric.__dict__)

    @classmethod
    def load_from_checkpoint(
        cls, checkpoint_path, map_location=None, hparams_file=None, strict=True, **kwargs
    ) -> MolAtomBondMPNN:
        submodules = {
            k: v
            for k, v in kwargs.items()
            if k
            in [
                "message_passing",
                "agg",
                "mol_predictor",
                "atom_predictor",
                "bond_predictor",
                "atom_constrainer",
                "bond_constrainer",
            ]
        }
        submodules, state_dict, hparams = cls._load(checkpoint_path, map_location, **submodules)
        kwargs.update(submodules)

        d = torch.load(checkpoint_path, map_location, weights_only=False)
        d["hyper_parameters"] = hparams
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
