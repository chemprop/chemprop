from abc import abstractmethod

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
    """
    A class that calibrates regression uncertainty models by applying
    a scaling value to the uncalibrated standard deviation, fitted by minimizing the
    negative log likelihood of a normal distribution around each prediction
    with scaling given by the uncalibrated variance. Method is described
    in https://arxiv.org/abs/1905.11659.
    """
    def fit(self, preds: Tensor, uncs: Tensor, targets: Tensor, mask: Tensor) -> None:
        ...
        return

    def apply(self, preds: Tensor, uncs: Tensor) -> tuple[Tensor, Tensor]:
        ...
        return


@UncertaintyCalibratorRegistry.register("zelikman-interval")
class ZelikmanCalibrator(UncertaintyCalibrator):
    """
    A calibrator for regression datasets that does not depend on a particular probability
    function form. Designed to be used with interval output. Uses the "CRUDE" method as
    described in https://arxiv.org/abs/2005.12496. As implemented here, the interval
    bounds are constrained to be symmetrical, though this is not required in the source method.
    The probability density to be used for NLL evaluator for the zelikman interval method is
    approximated here as a histogram function.
    """
    def fit(self, preds: Tensor, uncs: Tensor, targets: Tensor, mask: Tensor) -> None:
        ...
        return

    def apply(self, preds: Tensor, uncs: Tensor) -> tuple[Tensor, Tensor]:
        ...
        return


@UncertaintyCalibratorRegistry.register("mve-weighting")
class MVEWeightingCalibrator(UncertaintyCalibrator):
    """
    A method of calibration for models that have ensembles of individual models that
    make variance predictions. Minimizes the negative log likelihood for the
    predictions versus the targets by applying a weighted average across the
    variance predictions of the ensemble. Discussed in https://doi.org/10.1186/s13321-021-00551-x.
    """
    def fit(self, preds: Tensor, uncs: Tensor, targets: Tensor, mask: Tensor) -> None:
        ...
        return

    def apply(self, preds: Tensor, uncs: Tensor) -> tuple[Tensor, Tensor]:
        ...
        return


@UncertaintyCalibratorRegistry.register("platt")
class PlattCalibrator(UncertaintyCalibrator):
    def fit(self, preds: Tensor, uncs: Tensor, targets: Tensor, mask: Tensor) -> None:
        ...
        return

    def apply(self, preds: Tensor, uncs: Tensor) -> tuple[Tensor, Tensor]:
        ...
        return


@UncertaintyCalibratorRegistry.register("conformal-multilabel")
class ConformalMultilabelCalibrator(UncertaintyCalibrator):
    def fit(self, preds: Tensor, uncs: Tensor, targets: Tensor, mask: Tensor) -> None:
        ...
        return

    def apply(self, preds: Tensor, uncs: Tensor) -> tuple[Tensor, Tensor]:
        ...
        return


@UncertaintyCalibratorRegistry.register("conformal-multiclass")
class ConformalMulticlassCalibrator(UncertaintyCalibrator):
    def fit(self, preds: Tensor, uncs: Tensor, targets: Tensor, mask: Tensor) -> None:
        ...
        return

    def apply(self, preds: Tensor, uncs: Tensor) -> tuple[Tensor, Tensor]:
        ...
        return


@UncertaintyCalibratorRegistry.register("conformal-adaptive")
class ConformalAdaptiveMulticlassCalibrator(UncertaintyCalibrator):
    def fit(self, preds: Tensor, uncs: Tensor, targets: Tensor, mask: Tensor) -> None:
        ...
        return

    def apply(self, preds: Tensor, uncs: Tensor) -> tuple[Tensor, Tensor]:
        ...
        return


@UncertaintyCalibratorRegistry.register("conformal-regression")
class ConformalRegressionCalibrator(UncertaintyCalibrator):
    def fit(self, preds: Tensor, uncs: Tensor, targets: Tensor, mask: Tensor) -> None:
        ...
        return

    def apply(self, preds: Tensor, uncs: Tensor) -> tuple[Tensor, Tensor]:
        ...
        return


@UncertaintyCalibratorRegistry.register("conformal-quantile-regression")
class ConformalQuantileRegressionCalibrator(UncertaintyCalibrator):
    def fit(self, preds: Tensor, uncs: Tensor, targets: Tensor, mask: Tensor) -> None:
        ...
        return

    def apply(self, preds: Tensor, uncs: Tensor) -> tuple[Tensor, Tensor]:
        ...
        return


@UncertaintyCalibratorRegistry.register("isotonic")
class IsotonicCalibrator(UncertaintyCalibrator):
    def fit(self, preds: Tensor, uncs: Tensor, targets: Tensor, mask: Tensor) -> None:
        ...
        return

    def apply(self, preds: Tensor, uncs: Tensor) -> tuple[Tensor, Tensor]:
        ...
        return


@UncertaintyCalibratorRegistry.register("isotonic-multiclass")
class IsotonicMulticlassCalibrator(UncertaintyCalibrator):
    def fit(self, preds: Tensor, uncs: Tensor, targets: Tensor, mask: Tensor) -> None:
        ...
        return

    def apply(self, preds: Tensor, uncs: Tensor) -> tuple[Tensor, Tensor]:
        ...
        return
