from abc import abstractmethod
from dataclasses import dataclass

import torch
from torch import Tensor
from torchmetrics import functional as F
from torchmetrics.utilities.compute import auc

from chemprop.nn.loss import (
    BCELoss,
    BinaryMCCLoss,
    CrossEntropyLoss,
    LossFunction,
    MSELoss,
    MulticlassMCCLoss,
    SIDLoss,
    WassersteinLoss,
)
from chemprop.utils.registry import ClassRegistry

__all__ = [
    "Metric",
    "MetricRegistry",
    "ThresholdedMixin",
    "MAEMetric",
    "MSEMetric",
    "RMSEMetric",
    "BoundedMixin",
    "BoundedMAEMetric",
    "BoundedMSEMetric",
    "BoundedRMSEMetric",
    "R2Metric",
    "BinaryAUROCMetric",
    "BinaryAUPRCMetric",
    "BinaryAccuracyMetric",
    "BinaryF1Metric",
    "BCEMetric",
    "CrossEntropyMetric",
    "BinaryMCCMetric",
    "MulticlassMCCMetric",
    "SIDMetric",
    "WassersteinMetric",
]


class Metric(LossFunction):
    """
    Parameters
    ----------
    task_weights :  ArrayLike = 1.0
        .. important::
            Ignored. Maintained for compatibility with :class:`~chemprop.nn.loss.LossFunction`
    """

    minimize: bool = True

    def forward(
        self,
        preds: Tensor,
        targets: Tensor,
        mask: Tensor,
        weights: Tensor,
        lt_mask: Tensor,
        gt_mask: Tensor,
    ):
        return self._calc_unreduced_loss(preds, targets, mask, lt_mask, gt_mask)[mask].mean()

    @abstractmethod
    def _calc_unreduced_loss(self, preds, targets, mask, lt_mask, gt_mask) -> Tensor:
        pass


MetricRegistry = ClassRegistry[Metric]()


@dataclass
class ThresholdedMixin:
    threshold: float | None = 0.5

    def extra_repr(self) -> str:
        return f"threshold={self.threshold}"


@MetricRegistry.register("mae")
class MAEMetric(Metric):
    def _calc_unreduced_loss(self, preds, targets, *args) -> Tensor:
        return (preds - targets).abs()


@MetricRegistry.register("mse")
class MSEMetric(MSELoss, Metric):
    pass


@MetricRegistry.register("rmse")
class RMSEMetric(MSEMetric):
    def forward(
        self,
        preds: Tensor,
        targets: Tensor,
        mask: Tensor,
        weights: Tensor,
        lt_mask: Tensor,
        gt_mask: Tensor,
    ):
        squared_errors = super()._calc_unreduced_loss(preds, targets, mask, lt_mask, gt_mask)

        return squared_errors[mask].mean().sqrt()


class BoundedMixin:
    def _calc_unreduced_loss(self, preds, targets, mask, lt_mask, gt_mask) -> Tensor:
        preds = torch.where((preds < targets) & lt_mask, targets, preds)
        preds = torch.where((preds > targets) & gt_mask, targets, preds)

        return super()._calc_unreduced_loss(preds, targets, mask, lt_mask, gt_mask)


@MetricRegistry.register("bounded-mae")
class BoundedMAEMetric(MAEMetric, BoundedMixin):
    pass


@MetricRegistry.register("bounded-mse")
class BoundedMSEMetric(MSEMetric, BoundedMixin):
    pass


@MetricRegistry.register("bounded-rmse")
class BoundedRMSEMetric(RMSEMetric, BoundedMixin):
    pass


@MetricRegistry.register("r2")
class R2Metric(Metric):
    minimize = False

    def forward(self, preds: Tensor, targets: Tensor, mask: Tensor, *args, **kwargs):
        return F.r2_score(preds[mask], targets[mask])


@MetricRegistry.register("roc")
class BinaryAUROCMetric(Metric):
    minimize = False

    def forward(self, preds: Tensor, targets: Tensor, mask: Tensor, *args, **kwargs):
        return self._calc_unreduced_loss(preds, targets, mask)

    def _calc_unreduced_loss(self, preds, targets, mask, *args) -> Tensor:
        return F.auroc(preds[mask], targets[mask].long(), task="binary")


@MetricRegistry.register("prc")
class BinaryAUPRCMetric(Metric):
    minimize = False

    def forward(self, preds: Tensor, targets: Tensor, *args, **kwargs):
        p, r, _ = F.precision_recall_curve(preds, targets.long(), task="binary")
        return auc(r, p)


@MetricRegistry.register("accuracy")
class BinaryAccuracyMetric(Metric, ThresholdedMixin):
    minimize = False

    def forward(self, preds: Tensor, targets: Tensor, mask: Tensor, *args, **kwargs):
        return F.accuracy(
            preds[mask], targets[mask].long(), threshold=self.threshold, task="binary"
        )


@MetricRegistry.register("f1")
class BinaryF1Metric(Metric, ThresholdedMixin):
    minimize = False

    def forward(self, preds: Tensor, targets: Tensor, mask: Tensor, *args, **kwargs):
        return F.f1_score(
            preds[mask], targets[mask].long(), threshold=self.threshold, task="binary"
        )


@MetricRegistry.register("bce")
class BCEMetric(BCELoss, Metric):
    pass


@MetricRegistry.register("ce")
class CrossEntropyMetric(CrossEntropyLoss, Metric):
    pass


@MetricRegistry.register("binary-mcc")
class BinaryMCCMetric(BinaryMCCLoss, Metric):
    pass


@MetricRegistry.register("multiclass-mcc")
class MulticlassMCCMetric(MulticlassMCCLoss, Metric):
    pass


@MetricRegistry.register("sid")
class SIDMetric(SIDLoss, Metric):
    pass


@MetricRegistry.register("wasserstein")
class WassersteinMetric(WassersteinLoss, Metric):
    pass
