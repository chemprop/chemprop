from abc import abstractmethod

from scipy.optimize import fmin
import torch
from torch import Tensor

from chemprop.utils.registry import ClassRegistry


class UncertaintyCalibrator:
    @abstractmethod
    def fit(self, preds: Tensor, uncs: Tensor, targets: Tensor, mask: Tensor) -> None:
        """
        Fit calibration method for the calibration data.
        """

    @abstractmethod
    def apply(self, preds: Tensor, uncs: Tensor) -> tuple[Tensor, Tensor]:
        """
        Take in predictions and uncertainty parameters from a model and apply the calibration method using fitted parameters.
        """


UncertaintyCalibratorRegistry = ClassRegistry[UncertaintyCalibrator]()


@UncertaintyCalibratorRegistry.register("zscaling")
class ZScalingCalibrator(UncertaintyCalibrator):
    def fit(self, preds: Tensor, uncs: Tensor, targets: Tensor, mask: Tensor) -> None:
        ...

    def apply(self, preds: Tensor, uncs: Tensor) -> tuple[Tensor, Tensor]:
        ...
        return


@UncertaintyCalibratorRegistry.register("tscaling")
class TScalingCalibrator(UncertaintyCalibrator):
    def fit(self, preds: Tensor, uncs: Tensor, targets: Tensor, mask: Tensor) -> None:
        ...

    def apply(self, preds: Tensor, uncs: Tensor) -> tuple[Tensor, Tensor]:
        ...
        return


@UncertaintyCalibratorRegistry.register("zelikman-interval")
class ZelikmanCalibrator(UncertaintyCalibrator):
    def fit(self, preds: Tensor, uncs: Tensor, targets: Tensor, mask: Tensor) -> None:
        ...

    def apply(self, preds: Tensor, uncs: Tensor) -> tuple[Tensor, Tensor]:
        ...
        return


@UncertaintyCalibratorRegistry.register("mve-weighting")
class MVEWeightingCalibrator(UncertaintyCalibrator):
    def fit(self, preds: Tensor, uncs: Tensor, targets: Tensor, mask: Tensor) -> None:
        ...

    def apply(self, preds: Tensor, uncs: Tensor) -> tuple[Tensor, Tensor]:
        ...
        return


@UncertaintyCalibratorRegistry.register("platt")
class PlattCalibrator(UncertaintyCalibrator):
    """A calibration method for classification datasets based on the Platt scaling algorithm, as
    discussed in [guo2017]_ and [platt1999]_. In Platt's paper, he suggests using the number of
    positive and negative examples in the dataset used to train the model to adjust the value of
    target probabilities used to fit the parameters.

    References
    ----------
    .. [guo2017] Guo, C.; Pleiss, G.; Sun, Y.; Weinberger, K. Q. "On calibration of modern neural
        networks". ICML, 2017. https://arxiv.org/abs/1706.04599
    .. [platt1999] Platt, J.. "Probabilistic Outputs for Support Vector Machines and Comparisons to
        Regularized Likelihood Methods." Adv. Large Margin Classif. 1999, 10 (3), 61–74.
    """
    def fit(self, preds: Tensor, uncs: Tensor, targets: Tensor, mask: Tensor, training_targets: None | Tensor = None) -> None:
        if torch.any((targets[mask] != 0) & (targets[mask] != 1)):
            raise ValueError(
                "Platt scaling is only implemented for binary classification tasks! Input tensor "
                "must contain only 0's and 1's."
            )

        if training_targets is not None:
            n_negative_examples = (training_targets == 0).sum(dim=0)
            n_positive_examples = (training_targets == 1).sum(dim=0)

            negative_target_bayes_MAP = (1 / (n_negative_examples + 2)).expand_as(targets)
            positive_target_bayes_MAP = (
                (n_positive_examples + 1) / (n_positive_examples + 2)
            ).expand_as(targets)

            targets = targets.float()
            targets[targets == 0] = negative_target_bayes_MAP[targets == 0]
            targets[targets == 1] = positive_target_bayes_MAP[targets == 1]

        self.a = []
        self.b = []
        for j in range(preds.shape[1]):
            mask_j = mask[:, j]
            preds_j = preds[:, j][mask_j]
            targets_j = targets[:, j][mask_j]

            # if is_atom_bond_targets: # Not yet implemented

            def objective(parameters):
                a = parameters[0]
                b = parameters[1]
                scaled_preds = torch.sigmoid(a * torch.logit(preds_j) + b)
                nll = -1 * torch.sum(
                    targets_j * torch.log(scaled_preds)
                    + (1 - targets_j) * torch.log(1 - scaled_preds)
                )
                return nll.item()

            a_j, b_j = fmin(objective, x0=[1, 0], disp=False)
            self.a.append(a_j)
            self.a.append(b_j)

        self.a = torch.tensor(self.a)
        self.b = torch.tensor(self.b)

    def apply(self, preds: Tensor, uncs: Tensor) -> tuple[Tensor, Tensor]:
        return torch.sigmoid(self.a * torch.logit(preds) + self.b)


@UncertaintyCalibratorRegistry.register("conformal-multilabel")
class ConformalMultilabelCalibrator(UncertaintyCalibrator):
    def fit(self, preds: Tensor, uncs: Tensor, targets: Tensor, mask: Tensor) -> None:
        ...

    def apply(self, preds: Tensor, uncs: Tensor) -> tuple[Tensor, Tensor]:
        ...
        return


@UncertaintyCalibratorRegistry.register("conformal-multiclass")
class ConformalMulticlassCalibrator(UncertaintyCalibrator):
    def fit(self, preds: Tensor, uncs: Tensor, targets: Tensor, mask: Tensor) -> None:
        ...

    def apply(self, preds: Tensor, uncs: Tensor) -> tuple[Tensor, Tensor]:
        ...
        return


@UncertaintyCalibratorRegistry.register("conformal-adaptive")
class ConformalAdaptiveMulticlassCalibrator(UncertaintyCalibrator):
    def fit(self, preds: Tensor, uncs: Tensor, targets: Tensor, mask: Tensor) -> None:
        ...

    def apply(self, preds: Tensor, uncs: Tensor) -> tuple[Tensor, Tensor]:
        ...
        return


@UncertaintyCalibratorRegistry.register("conformal-regression")
class ConformalRegressionCalibrator(UncertaintyCalibrator):
    def fit(self, preds: Tensor, uncs: Tensor, targets: Tensor, mask: Tensor) -> None:
        ...

    def apply(self, preds: Tensor, uncs: Tensor) -> tuple[Tensor, Tensor]:
        ...
        return


@UncertaintyCalibratorRegistry.register("conformal-quantile-regression")
class ConformalQuantileRegressionCalibrator(UncertaintyCalibrator):
    def fit(self, preds: Tensor, uncs: Tensor, targets: Tensor, mask: Tensor) -> None:
        ...

    def apply(self, preds: Tensor, uncs: Tensor) -> tuple[Tensor, Tensor]:
        ...
        return


@UncertaintyCalibratorRegistry.register("isotonic")
class IsotonicCalibrator(UncertaintyCalibrator):
    def fit(self, preds: Tensor, uncs: Tensor, targets: Tensor, mask: Tensor) -> None:
        ...

    def apply(self, preds: Tensor, uncs: Tensor) -> tuple[Tensor, Tensor]:
        ...
        return


@UncertaintyCalibratorRegistry.register("isotonic-multiclass")
class IsotonicMulticlassCalibrator(UncertaintyCalibrator):
    def fit(self, preds: Tensor, uncs: Tensor, targets: Tensor, mask: Tensor) -> None:
        ...

    def apply(self, preds: Tensor, uncs: Tensor) -> tuple[Tensor, Tensor]:
        ...
        return
