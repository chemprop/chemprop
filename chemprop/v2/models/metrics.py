from abc import ABC, abstractmethod

import torch
from torch import Tensor
from torch.nn.functional import cross_entropy, binary_cross_entropy_with_logits
from torchmetrics import functional as F

from chemprop.v2.utils import ClassFactory

MetricFactory = ClassFactory()


class Metric(ABC):
    """A `Metric` is like a loss function, but it calculates only a single scalar for the entire
    batch.

    NOTE(degraff): this can probably be rewritten to subclass from `LossFunction`
    """

    minimized: bool

    def __call__(self, preds: Tensor, targets: Tensor, mask: Tensor, **kwargs) -> Tensor:
        return self.forward(preds, targets, mask, **kwargs)
    
    @abstractmethod
    def forward(self, preds: Tensor, targets: Tensor, mask: Tensor, **kwargs) -> Tensor:
        pass


class MinimizedMetric(Metric):
    minimized = True


class MaximizedMetric(Metric):
    minimized = False


class ThresholdedMixin:
    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold


@MetricFactory.register("mae")
class MAEMetric(MinimizedMetric):
    def forward(self, preds: Tensor, targets: Tensor, mask: Tensor, **kwargs) -> Tensor:
        return (preds - targets)[mask].abs().mean()


@MetricFactory.register("mse")
class MSEMetric(MinimizedMetric):
    def forward(self, preds: Tensor, targets: Tensor, mask: Tensor, **kwargs) -> Tensor:
        return (preds - targets)[mask].square().mean()


@MetricFactory.register("rmse")
class RMSEMetric(MSEMetric):
    def forward(self, preds: Tensor, targets: Tensor, mask: Tensor, **kwargs) -> Tensor:
        return super().forward(preds, targets, mask).sqrt()


class BoundedMixin:
    def __call__(
        self,
        preds: Tensor,
        targets: Tensor,
        mask: Tensor,
        lt_mask: Tensor,
        gt_mask: Tensor,
        **kwargs,
    ) -> Tensor:
        preds = self.bound(preds, targets, lt_mask, gt_mask)

        return super().__call__(preds, targets, mask)

    def bound(self, preds, targets, lt_mask, gt_mask) -> Tensor:
        preds[preds < targets & lt_mask] = targets
        preds[preds > targets & gt_mask] = targets

        return preds


@MetricFactory.register("bounded-mae")
class BoundedMAEMetric(BoundedMixin, MAEMetric):
    pass


@MetricFactory.register("bounded-mse")
class BoundedMSEMetric(BoundedMixin, MSEMetric):
    pass


@MetricFactory.register("bounded-rmse")
class BoundedRMSEMetric(BoundedMixin, RMSEMetric):
    pass


@MetricFactory.register("r2")
class R2Metric(MaximizedMetric):
    def forward(self, preds: Tensor, targets: Tensor, mask: Tensor, **kwargs) -> Tensor:
        return F.r2_score(preds[mask], targets[mask])


@MetricFactory.register("roc")
class AUROCMetric(MaximizedMetric):
    def forward(self, preds: Tensor, targets: Tensor, mask: Tensor, **kwargs) -> Tensor:
        return F.auroc(preds[mask], targets[mask].long())


@MetricFactory.register("prc")
class AUPRCMetric(MaximizedMetric):
    def forward(self, preds: Tensor, targets: Tensor, mask: Tensor, **kwargs) -> Tensor:
        p, r, _ = F.precision_recall(preds, targets.long())

        return F.auc(r, p)


@MetricFactory.register("accuracy")
class AccuracyMetric(MaximizedMetric, ThresholdedMixin):
    def forward(self, preds: Tensor, targets: Tensor, mask: Tensor, **kwargs) -> Tensor:
        return F.accuracy(preds[mask], targets[mask].long(), threshold=self.threshold)


@MetricFactory.register("f1")
class F1Metric(MaximizedMetric):
    def forward(self, preds: Tensor, targets: Tensor, mask: Tensor, **kwargs) -> Tensor:
        return F.f1_score(preds[mask], targets[mask].long(), threshold=self.threshold)


@MetricFactory.register("bce")
class BCEMetric(MaximizedMetric):
    def forward(self, preds: Tensor, targets: Tensor, mask: Tensor, **kwargs) -> Tensor:
        return binary_cross_entropy_with_logits(preds[mask], targets[mask].long())


@MetricFactory.register("mcc")
class MCCMetric(MaximizedMetric):
    """NOTE(degraff): don't think this works rn"""
    def __init__(self, n_classes: int, threshold: float = 0.5, **kwargs) -> Tensor:
        self.n_classes = n_classes
        self.threshold = threshold

    def forward(self, preds: Tensor, targets: Tensor, mask: Tensor, **kwargs) -> Tensor:
        return F.matthews_corrcoef(
            preds[mask], targets[mask].long(), self.n_classes, self.threshold
        )


@MetricFactory.register("xent")
class CrossEntropyMetric(MinimizedMetric):
    def forward(self, preds: Tensor, targets: Tensor, mask: Tensor, **kwargs) -> Tensor:
        return cross_entropy(preds[mask], targets[mask].long())


@MetricFactory.register("spectral-sid")
class SIDMetric(MinimizedMetric, ThresholdedMixin):
    def forward(self, preds: Tensor, targets: Tensor, mask: Tensor, **kwargs) -> Tensor:
        if self.threshold is not None:
            preds = preds.clamp(min=self.threshold)

        preds_norm = preds / (preds * mask).sum(1, keepdim=True)

        targets = targets.masked_fill(~mask, 1)
        preds_norm = preds_norm.masked_fill(~mask, 1)

        return (
            torch.log(preds_norm / targets) * preds_norm + torch.log(targets / preds_norm) * targets
        )[mask].mean()


@MetricFactory.register("wasserstein-sid")
class WassersteinMetric(MinimizedMetric, ThresholdedMixin):
    def forward(self, preds: Tensor, targets: Tensor, mask: Tensor, **kwargs) -> Tensor:
        if self.threshold is not None:
            preds = preds.clamp(min=self.threshold)

        preds_norm = preds / (preds * mask).sum(1, keepdim=True)

        return torch.abs(targets.cumsum(1) - preds_norm.cumsum(1))[mask].mean()
