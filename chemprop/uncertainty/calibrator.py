from abc import ABC, abstractmethod
from typing import Self

from torch import Tensor

from chemprop.utils.registry import ClassRegistry


class UncertaintyCalibrator(ABC):
    """
    A class for calibrating the predicted uncertainties.
    """

    @abstractmethod
    def fit(self, preds: Tensor, uncs: Tensor, targets: Tensor, mask: Tensor) -> Self:
        """
        Fit calibration method for the calibration data.

        .. note::
            The `preds` variable is required only for regression tasks.

            The `uncs` variable holds different values depending on the task:

            - Regression tasks: `uncs` represents the predicted variance.
            - Binary classification: `uncs` is the predicted probability of class 1.
            - Multiclass classification: `uncs` contains the predicted probabilities for each class.

        Parameters
        ----------
        preds: Tensor
            the predictions for regression tasks. It is a tensor of the shape of ``n x t``, where ``n`` is the number of input
            molecules/reactions, and ``t`` is the number of tasks.
        uncs: Tensor
            the predicted uncertainties of varying shape depending on the task type:

            * regression/binary classification: ``n x t``

            * multiclass classification: ``n x t x c``, where ``c`` is the number of classes
        targets: Tensor
            a tensor of the shape ``n x t``
        mask: Tensor
            a tensor of the shape ``n x t`` indicating whether the given values should be used in the fitting

        Returns
        -------
        self : UncertaintyCalibrator
            the fitted calibrator
        """

    @abstractmethod
    def apply(self, uncs: Tensor) -> Tensor:
        """
        Apply this calibrator to the input uncertainties.

        Parameters
        ----------
        uncs: Tensor
            a tensor containinig uncalibrated uncertainties

        Returns
        -------
        Tensor
            the calibrated uncertainties
        """


UncertaintyCalibratorRegistry = ClassRegistry[UncertaintyCalibrator]()


@UncertaintyCalibratorRegistry.register("zscaling")
class ZScalingCalibrator(UncertaintyCalibrator):
    def fit(self, preds: Tensor, uncs: Tensor, targets: Tensor, mask: Tensor) -> Self:
        ...
        return self

    def apply(self, uncs: Tensor) -> Tensor:
        ...
        return


@UncertaintyCalibratorRegistry.register("tscaling")
class TScalingCalibrator(UncertaintyCalibrator):
    def fit(self, preds: Tensor, uncs: Tensor, targets: Tensor, mask: Tensor) -> Self:
        ...
        return self

    def apply(self, uncs: Tensor) -> Tensor:
        ...
        return


@UncertaintyCalibratorRegistry.register("zelikman-interval")
class ZelikmanCalibrator(UncertaintyCalibrator):
    def fit(self, preds: Tensor, uncs: Tensor, targets: Tensor, mask: Tensor) -> Self:
        ...
        return self

    def apply(self, uncs: Tensor) -> Tensor:
        ...
        return


@UncertaintyCalibratorRegistry.register("mve-weighting")
class MVEWeightingCalibrator(UncertaintyCalibrator):
    def fit(self, preds: Tensor, uncs: Tensor, targets: Tensor, mask: Tensor) -> Self:
        ...
        return self

    def apply(self, uncs: Tensor) -> Tensor:
        ...
        return


@UncertaintyCalibratorRegistry.register("platt")
class PlattCalibrator(UncertaintyCalibrator):
    def fit(self, preds: Tensor, uncs: Tensor, targets: Tensor, mask: Tensor) -> Self:
        ...
        return self

    def apply(self, uncs: Tensor) -> Tensor:
        ...
        return


@UncertaintyCalibratorRegistry.register("conformal-multilabel")
class ConformalMultilabelCalibrator(UncertaintyCalibrator):
    def fit(self, preds: Tensor, uncs: Tensor, targets: Tensor, mask: Tensor) -> Self:
        ...
        return self

    def apply(self, uncs: Tensor) -> Tensor:
        ...
        return


@UncertaintyCalibratorRegistry.register("conformal-multiclass")
class ConformalMulticlassCalibrator(UncertaintyCalibrator):
    def fit(self, preds: Tensor, uncs: Tensor, targets: Tensor, mask: Tensor) -> Self:
        ...
        return self

    def apply(self, uncs: Tensor) -> Tensor:
        ...
        return


@UncertaintyCalibratorRegistry.register("conformal-adaptive")
class ConformalAdaptiveMulticlassCalibrator(UncertaintyCalibrator):
    def fit(self, preds: Tensor, uncs: Tensor, targets: Tensor, mask: Tensor) -> Self:
        ...
        return self

    def apply(self, uncs: Tensor) -> Tensor:
        ...
        return


@UncertaintyCalibratorRegistry.register("conformal-regression")
class ConformalRegressionCalibrator(UncertaintyCalibrator):
    def fit(self, preds: Tensor, uncs: Tensor, targets: Tensor, mask: Tensor) -> Self:
        ...
        return self

    def apply(self, uncs: Tensor) -> Tensor:
        ...
        return


@UncertaintyCalibratorRegistry.register("conformal-quantile-regression")
class ConformalQuantileRegressionCalibrator(UncertaintyCalibrator):
    def fit(self, preds: Tensor, uncs: Tensor, targets: Tensor, mask: Tensor) -> Self:
        ...
        return self

    def apply(self, uncs: Tensor) -> Tensor:
        ...
        return


@UncertaintyCalibratorRegistry.register("isotonic")
class IsotonicCalibrator(UncertaintyCalibrator):
    def fit(self, preds: Tensor, uncs: Tensor, targets: Tensor, mask: Tensor) -> Self:
        ...
        return self

    def apply(self, uncs: Tensor) -> Tensor:
        ...
        return


@UncertaintyCalibratorRegistry.register("isotonic-multiclass")
class IsotonicMulticlassCalibrator(UncertaintyCalibrator):
    def fit(self, preds: Tensor, uncs: Tensor, targets: Tensor, mask: Tensor) -> Self:
        ...
        return self

    def apply(self, uncs: Tensor) -> Tensor:
        ...
        return
