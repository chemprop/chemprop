from __future__ import annotations

import io
import logging
import traceback
from typing import Iterable, TypeAlias

from lightning import pytorch as pl
import torch
from torch import Tensor, nn, optim

from chemprop.data import BatchMolGraph, MulticomponentTrainingBatch, TrainingBatch
from chemprop.nn import Aggregation, ChempropMetric, MessagePassing, Predictor
from chemprop.nn.transforms import ScaleTransform
from chemprop.schedulers import build_NoamLike_LRSched
from chemprop.utils.registry import Factory

logger = logging.getLogger(__name__)

BatchType: TypeAlias = TrainingBatch | MulticomponentTrainingBatch


class MPNN(pl.LightningModule):
    r"""An :class:`MPNN` is a sequence of message passing layers, an aggregation routine, and a
    predictor routine.

    The first two modules calculate learned fingerprints from an input molecule or
    reaction graph, and the final module takes these learned fingerprints as input to calculate a
    final prediction. I.e., the following operation:

    .. math::
        \mathtt{MPNN}(\mathcal{G}) =
            \mathtt{predictor}(\mathtt{agg}(\mathtt{message\_passing}(\mathcal{G})))

    The full model is trained end-to-end.

    Parameters
    ----------
    message_passing : MessagePassing
        the message passing block to use to calculate learned fingerprints
    agg : Aggregation
        the aggregation operation to use during molecule-level prediction
    predictor : Predictor
        the function to use to calculate the final prediction
    batch_norm : bool, default=False
        if `True`, apply batch normalization to the output of the aggregation operation
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
        the predictor function
    """

    def __init__(
        self,
        message_passing: MessagePassing,
        agg: Aggregation,
        predictor: Predictor,
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
        self.save_hyperparameters(ignore=["X_d_transform", "message_passing", "agg", "predictor"])
        self.hparams["X_d_transform"] = X_d_transform
        self.hparams.update(
            {
                "message_passing": message_passing.hparams,
                "agg": agg.hparams,
                "predictor": predictor.hparams,
            }
        )

        self.message_passing = message_passing
        self.agg = agg
        self.bn = nn.BatchNorm1d(self.message_passing.output_dim) if batch_norm else nn.Identity()
        self.predictor = predictor

        self.X_d_transform = X_d_transform if X_d_transform is not None else nn.Identity()

        self.metrics = (
            nn.ModuleList([*metrics, self.criterion.clone()])
            if metrics
            else nn.ModuleList([self.predictor._T_default_metric(), self.criterion.clone()])
        )

        self.warmup_epochs = warmup_epochs
        self.init_lr = init_lr
        self.max_lr = max_lr
        self.final_lr = final_lr

    @property
    def output_dim(self) -> int:
        return self.predictor.output_dim

    @property
    def n_tasks(self) -> int:
        return self.predictor.n_tasks

    @property
    def n_targets(self) -> int:
        return self.predictor.n_targets

    @property
    def criterion(self) -> ChempropMetric:
        return self.predictor.criterion

    def fingerprint(
        self, bmg: BatchMolGraph, V_d: Tensor | None = None, X_d: Tensor | None = None
    ) -> Tensor:
        """the learned fingerprints for the input molecules"""
        H_v = self.message_passing(bmg, V_d)
        H = self.agg(H_v, bmg.batch)
        H = self.bn(H)

        return H if X_d is None else torch.cat((H, self.X_d_transform(X_d)), dim=1)

    def encoding(
        self, bmg: BatchMolGraph, V_d: Tensor | None = None, X_d: Tensor | None = None, i: int = -1
    ) -> Tensor:
        """Calculate the :attr:`i`-th hidden representation"""
        return self.predictor.encode(self.fingerprint(bmg, V_d, X_d), i)

    def forward(
        self, bmg: BatchMolGraph, V_d: Tensor | None = None, X_d: Tensor | None = None
    ) -> Tensor:
        """Generate predictions for the input molecules/reactions"""
        return self.predictor(self.fingerprint(bmg, V_d, X_d))

    def training_step(self, batch: BatchType, batch_idx):
        batch_size = self.get_batch_size(batch)
        bmg, V_d, X_d, targets, weights, lt_mask, gt_mask = batch

        mask = targets.isfinite()
        targets = targets.nan_to_num(nan=0.0)

        Z = self.fingerprint(bmg, V_d, X_d)
        preds = self.predictor.train_step(Z)
        l = self.criterion(preds, targets, mask, weights, lt_mask, gt_mask)

        self.log("train_loss", self.criterion, batch_size=batch_size, prog_bar=True, on_epoch=True)

        return l

    def on_validation_model_eval(self) -> None:
        self.eval()
        self.message_passing.V_d_transform.train()
        self.message_passing.graph_transform.train()
        self.X_d_transform.train()
        self.predictor.output_transform.train()

    def validation_step(self, batch: BatchType, batch_idx: int = 0):
        self._evaluate_batch(batch, "val")

        batch_size = self.get_batch_size(batch)
        bmg, V_d, X_d, targets, weights, lt_mask, gt_mask = batch

        mask = targets.isfinite()
        targets = targets.nan_to_num(nan=0.0)

        Z = self.fingerprint(bmg, V_d, X_d)
        preds = self.predictor.train_step(Z)
        self.metrics[-1](preds, targets, mask, weights, lt_mask, gt_mask)
        self.log("val_loss", self.metrics[-1], batch_size=batch_size, prog_bar=True)

    def test_step(self, batch: BatchType, batch_idx: int = 0):
        self._evaluate_batch(batch, "test")

    def _evaluate_batch(self, batch: BatchType, label: str) -> None:
        batch_size = self.get_batch_size(batch)
        bmg, V_d, X_d, targets, weights, lt_mask, gt_mask = batch

        mask = targets.isfinite()
        targets = targets.nan_to_num(nan=0.0)
        preds = self(bmg, V_d, X_d)
        weights = torch.ones_like(weights)

        if self.predictor.n_targets > 1:
            preds = preds[..., 0]

        for m in self.metrics[:-1]:
            m.update(preds, targets, mask, weights, lt_mask, gt_mask)
            self.log(f"{label}/{m.alias}", m, batch_size=batch_size)

    def predict_step(self, batch: BatchType, batch_idx: int, dataloader_idx: int = 0) -> Tensor:
        bmg, V_d, X_d, *_ = batch

        return self(bmg, V_d, X_d)

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

    def get_batch_size(self, batch: TrainingBatch) -> int:
        return len(batch[0])

    @classmethod
    def _load(cls, path, map_location, **submodules):
        try:
            d = torch.load(path, map_location, weights_only=False)
        except AttributeError:
            logger.error(
                f"{traceback.format_exc()}\nModel loading failed (full stacktrace above)! It is possible this checkpoint was generated in v2.0 and needs to be converted to v2.1\n Please run 'chemprop convert --conversion v2_0_to_v2_1 -i {path}' and load the converted checkpoint."
            )

        try:
            hparams = d["hyper_parameters"]
            state_dict = d["state_dict"]
        except KeyError:
            raise KeyError(f"Could not find hyper parameters and/or state dict in {path}.")

        if hparams["metrics"] is not None:
            hparams["metrics"] = [
                cls._rebuild_metric(metric)
                if not hasattr(metric, "_defaults")
                or (not torch.cuda.is_available() and metric.device.type != "cpu")
                else metric
                for metric in hparams["metrics"]
            ]

        if hparams["predictor"]["criterion"] is not None:
            metric = hparams["predictor"]["criterion"]
            if not hasattr(metric, "_defaults") or (
                not torch.cuda.is_available() and metric.device.type != "cpu"
            ):
                hparams["predictor"]["criterion"] = cls._rebuild_metric(metric)

        submodules |= {
            key: hparams[key].pop("cls")(**hparams[key])
            for key in ("message_passing", "agg", "predictor")
            if key not in submodules
        }

        return submodules, state_dict, hparams

    @classmethod
    def _add_metric_task_weights_to_state_dict(cls, state_dict, hparams):
        if "metrics.0.task_weights" not in state_dict:
            metrics = hparams["metrics"]
            n_metrics = len(metrics) if metrics is not None else 1
            for i_metric in range(n_metrics):
                state_dict[f"metrics.{i_metric}.task_weights"] = torch.tensor([[1.0]])
            state_dict[f"metrics.{i_metric + 1}.task_weights"] = state_dict[
                "predictor.criterion.task_weights"
            ]
        return state_dict

    @classmethod
    def _rebuild_metric(cls, metric):
        return Factory.build(metric.__class__, task_weights=metric.task_weights, **metric.__dict__)

    @classmethod
    def load_from_checkpoint(
        cls, checkpoint_path, map_location=None, hparams_file=None, strict=True, **kwargs
    ) -> MPNN:
        submodules = {
            k: v for k, v in kwargs.items() if k in ["message_passing", "agg", "predictor"]
        }
        submodules, state_dict, hparams = cls._load(checkpoint_path, map_location, **submodules)
        kwargs.update(submodules)

        state_dict = cls._add_metric_task_weights_to_state_dict(state_dict, hparams)
        d = torch.load(checkpoint_path, map_location, weights_only=False)
        d["state_dict"] = state_dict
        d["hyper_parameters"] = hparams
        buffer = io.BytesIO()
        torch.save(d, buffer)
        buffer.seek(0)

        return super().load_from_checkpoint(buffer, map_location, hparams_file, strict, **kwargs)

    @classmethod
    def load_from_file(cls, model_path, map_location=None, strict=True, **submodules) -> MPNN:
        submodules, state_dict, hparams = cls._load(model_path, map_location, **submodules)
        hparams.update(submodules)

        state_dict = cls._add_metric_task_weights_to_state_dict(state_dict, hparams)

        model = cls(**hparams)
        model.load_state_dict(state_dict, strict=strict)

        return model
