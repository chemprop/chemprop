from abc import abstractmethod
from dataclasses import dataclass

import torch
from torch import Tensor
from torchmetrics import functional as F

from chemprop.utils.registry import ClassRegistry
from chemprop.nn.loss import *

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
    "AUROCMetric",
    "AUPRCMetric",
    "AccuracyMetric",
    "F1Metric",
    "BCEMetric",
    "CrossEntropyMetric",
    "BinaryMCCMetric",
    "MulticlassMCCMetric",
    "SIDMetric",
    "WassersteinMetric",
]


class Metric(LossFunction):
    minimize: bool = True

    def __call__(
        self,
        preds: Tensor,
        targets: Tensor,
        mask: Tensor,
        w_s: Tensor,
        w_t: Tensor,
        lt_mask: Tensor,
        gt_mask: Tensor,
    ):
        return self.forward(preds, targets, mask, lt_mask, gt_mask)[mask].mean()

    @abstractmethod
    def forward(self, preds, targets, mask, lt_mask, gt_mask) -> Tensor:
        pass


MetricRegistry = ClassRegistry[Metric]()


@dataclass
class ThresholdedMixin:
    threshold: float | None = 0.5


@MetricRegistry.register("mae")
class MAEMetric(Metric):
    def forward(self, preds, targets, *args) -> Tensor:
        return (preds - targets).abs()


@MetricRegistry.register("mse")
class MSEMetric(MSELoss, Metric):
    pass


@MetricRegistry.register("rmse")
class RMSEMetric(MSEMetric):
    def __call__(
        self,
        preds: Tensor,
        targets: Tensor,
        mask: Tensor,
        w_s: Tensor,
        w_t: Tensor,
        lt_mask: Tensor,
        gt_mask: Tensor,
    ):
        return super().forward(preds, targets, mask, lt_mask, gt_mask)[mask].mean().sqrt()


class BoundedMixin:
    def forward(self, preds, targets, mask, lt_mask, gt_mask) -> Tensor:
        preds = torch.where((preds < targets) & lt_mask, targets, preds)
        preds = torch.where((preds > targets) & gt_mask, targets, preds)

        return super().forward(preds, targets, mask, lt_mask, gt_mask)


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

    def __call__(self, preds: Tensor, targets: Tensor, mask: Tensor, *args, **kwargs):
        return F.r2_score(preds[mask], targets[mask])


@MetricRegistry.register("roc")
class AUROCMetric(Metric):
    minimize = False

    def __init__(self, task: str) -> None:
        self.task = task

    def __call__(self, preds: Tensor, targets: Tensor, mask: Tensor, *args, **kwargs):
        return self.forward(preds, targets, mask)

    def forward(self, preds, targets, mask, *args) -> Tensor:
        return F.auroc(preds[mask], targets[mask].long(), task=self.task)


@MetricRegistry.register("prc")
class AUPRCMetric(Metric):
    minimize = False

    def __call__(self, preds: Tensor, targets: Tensor, *args, **kwargs):
        p, r, _ = F.precision_recall(preds, targets.long())

        return F.auc(r, p)


@MetricRegistry.register("accuracy")
class AccuracyMetric(Metric, ThresholdedMixin):
    minimize = False

    def __call__(self, preds: Tensor, targets: Tensor, mask: Tensor, *args, **kwargs):
        return F.accuracy(preds[mask], targets[mask].long(), threshold=self.threshold)


@MetricRegistry.register("f1")
class F1Metric(Metric):
    minimize = False

    def __call__(self, preds: Tensor, targets: Tensor, mask: Tensor, *args, **kwargs):
        return F.f1_score(preds[mask], targets[mask].long(), threshold=self.threshold)


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
