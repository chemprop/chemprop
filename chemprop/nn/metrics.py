from numpy.typing import ArrayLike
import torch
from torch import Tensor
import torchmetrics
from torchmetrics.utilities.compute import auc

from chemprop.nn.loss import MSE, BinaryMCCLoss, ChempropMetric, MulticlassMCCLoss
from chemprop.utils.registry import ClassRegistry

__all__ = [
    "MAE",
    "RMSE",
    "BinaryAccuracy",
    "BinaryAUPRC",
    "BinaryAUROC",
    "BinaryF1Score",
    "BinaryMCCMetric",
    "BoundedMAE",
    "BoundedMSE",
    "BoundedRMSE",
    "MetricRegistry",
    "MulticlassMCCMetric",
    "R2Score",
]


MetricRegistry = ClassRegistry[ChempropMetric]()


@MetricRegistry.register("mae")
class MAE(ChempropMetric):
    def _calc_unreduced_loss(self, preds, targets, *args) -> Tensor:
        return (preds - targets).abs()


@MetricRegistry.register("rmse")
class RMSE(MSE):
    def compute(self):
        return (self.total_loss / self.num_samples).sqrt()


class BoundedMixin:
    def _calc_unreduced_loss(self, preds, targets, mask, weights, lt_mask, gt_mask) -> Tensor:
        preds = torch.where((preds < targets) & lt_mask, targets, preds)
        preds = torch.where((preds > targets) & gt_mask, targets, preds)

        return super()._calc_unreduced_loss(preds, targets, mask, weights)


@MetricRegistry.register("bounded-mae")
class BoundedMAE(BoundedMixin, MAE):
    pass


@MetricRegistry.register("bounded-mse")
class BoundedMSE(BoundedMixin, MSE):
    pass


@MetricRegistry.register("bounded-rmse")
class BoundedRMSE(BoundedMixin, RMSE):
    pass


@MetricRegistry.register("r2")
class R2Score(torchmetrics.R2Score):
    def update(self, preds: Tensor, targets: Tensor, mask: Tensor, *args, **kwargs):
        super().update(preds[mask], targets[mask])


class ChempropClassificationMixin:
    def __init__(self, task_weights: ArrayLike = 1.0, **kwargs):
        """
        Parameters
        ----------
        task_weights :  ArrayLike = 1.0
            .. important::
                Ignored. Maintained for compatibility with :class:`ChempropMetric`
        """
        super().__init__()
        task_weights = torch.as_tensor(task_weights, dtype=torch.float).view(1, -1)
        self.register_buffer("task_weights", task_weights)

    def update(self, preds: Tensor, targets: Tensor, mask: Tensor, *args, **kwargs):
        super().update(preds[mask], targets[mask].long())


@MetricRegistry.register("roc")
class BinaryAUROC(ChempropClassificationMixin, torchmetrics.classification.BinaryAUROC):
    pass


@MetricRegistry.register("prc")
class BinaryAUPRC(
    ChempropClassificationMixin, torchmetrics.classification.BinaryPrecisionRecallCurve
):
    def compute(self) -> Tensor:
        p, r, _ = super().compute()
        return auc(r, p)


@MetricRegistry.register("accuracy")
class BinaryAccuracy(ChempropClassificationMixin, torchmetrics.classification.BinaryAccuracy):
    pass


@MetricRegistry.register("f1")
class BinaryF1Score(ChempropClassificationMixin, torchmetrics.classification.BinaryF1Score):
    pass


@MetricRegistry.register("binary-mcc")
class BinaryMCCMetric(BinaryMCCLoss):
    def compute(self):
        return 1 - super().compute()


@MetricRegistry.register("multiclass-mcc")
class MulticlassMCCMetric(MulticlassMCCLoss):
    def compute(self):
        return 1 - super().compute()
