from abc import abstractmethod
from dataclasses import dataclass

import torch
from torch import Tensor
from torchmetrics import functional as F

from chemprop.v2.utils.registry import ClassRegistry
from chemprop.v2.nn.loss import BCELoss, CrossEntropyLoss, LossFunction, MSELoss

MetricRegistry = ClassRegistry()


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
    def forward(self, *args, **kwargs) -> Tensor:
        return super().forward(*args, **kwargs).sqrt()


class BoundedMixin:
    def forward(self, preds, targets, mask, lt_mask, gt_mask) -> Tensor:
        preds[preds < targets & lt_mask] = targets
        preds[preds > targets & gt_mask] = targets

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

    def __call__(self, preds: Tensor, targets: Tensor, mask: Tensor, *args, **kwargs):
        return F.auroc(preds[mask], targets[mask].long())


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
    # def forward(self, preds, targets, mask, lt_mask, gt_mask) -> Tensor:
    #     return binary_cross_entropy_with_logits(preds[mask], targets[mask].long())


@MetricRegistry.register("ce")
class CrossEntropyMetric(CrossEntropyLoss, Metric):
    pass
    # def forward(self, preds, targets, mask, lt_mask, gt_mask) -> Tensor:
    #     return cross_entropy(preds[mask], targets[mask].long())


@MetricRegistry.register("mcc")
class MCCMetric(Metric):
    minimize = False
    """NOTE(degraff): don't think this works rn"""

    def __init__(self, n_classes: int, threshold: float = 0.5, *args) -> Tensor:
        self.n_classes = n_classes
        self.threshold = threshold

    def __call__(self, preds: Tensor, targets: Tensor, mask: Tensor, *args, **kwargs):
        return F.matthews_corrcoef(
            preds[mask], targets[mask].long(), self.n_classes, self.threshold
        )


@MetricRegistry.register("sid")
class SIDMetric(Metric, ThresholdedMixin):
    def forward(self, preds, targets, mask, *args) -> Tensor:
        preds = preds.clamp(min=self.threshold)
        preds_norm = preds / (preds * mask).sum(1, keepdim=True)

        targets = targets.masked_fill(~mask, 1)
        preds_norm = preds_norm.masked_fill(~mask, 1)

        return (
            torch.log(preds_norm / targets) * preds_norm + torch.log(targets / preds_norm) * targets
        )


@MetricRegistry.register("wasserstein")
class WassersteinMetric(Metric, ThresholdedMixin):
    def forward(self, preds: Tensor, targets, mask, *args) -> Tensor:
        preds = preds.clamp(min=self.threshold)
        preds_norm = preds / (preds * mask).sum(1, keepdim=True)

        return torch.abs(targets.cumsum(1) - preds_norm.cumsum(1))
