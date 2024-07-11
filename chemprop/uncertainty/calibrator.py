from abc import abstractmethod

from torch import Tensor

from chemprop.utils.registry import ClassRegistry


class UncertaintyCalibrator:
    @abstractmethod
    def calibrate(self, preds: Tensor, uncs: Tensor, targets: Tensor, mask: Tensor) -> Tensor:
        """
        Fit calibration method for the calibration data.
        """

    @abstractmethod
    def apply_calibration(self, preds: Tensor, uncs: Tensor) -> Tensor:
        """
        Take in predictions and uncertainty parameters from a model and apply the calibration method using fitted parameters.
        """


UncertaintyCalibratorRegistry = ClassRegistry[UncertaintyCalibrator]()


@UncertaintyCalibratorRegistry.register("zscaling")
class ZScalingCalibrator(UncertaintyCalibrator):
    def calibrate(self, preds, uncs, targets, mask) -> Tensor:
        ...
        return

    def apply_calibration(self, preds, uncs) -> Tensor:
        ...
        return


@UncertaintyCalibratorRegistry.register("tscaling")
class TScalingCalibrator(UncertaintyCalibrator):
    def calibrate(self, preds, uncs, targets, mask) -> Tensor:
        ...
        return

    def apply_calibration(self, preds, uncs) -> Tensor:
        ...
        return


@UncertaintyCalibratorRegistry.register("zelikman-interval")
class ZelikmanCalibrator(UncertaintyCalibrator):
    def calibrate(self, preds, uncs, targets, mask) -> Tensor:
        ...
        return

    def apply_calibration(self, preds, uncs) -> Tensor:
        ...
        return


@UncertaintyCalibratorRegistry.register("mve-weighting")
class MVEWeightingCalibrator(UncertaintyCalibrator):
    def calibrate(self, preds, uncs, targets, mask) -> Tensor:
        ...
        return

    def apply_calibration(self, preds, uncs) -> Tensor:
        ...
        return


@UncertaintyCalibratorRegistry.register("platt")
class PlattCalibrator(UncertaintyCalibrator):
    def calibrate(self, preds, uncs, targets, mask) -> Tensor:
        ...
        return

    def apply_calibration(self, preds, uncs) -> Tensor:
        ...
        return


@UncertaintyCalibratorRegistry.register("conformal-multilabel")
class ConformalMultilabelCalibrator(UncertaintyCalibrator):
    def calibrate(self, preds, uncs, targets, mask) -> Tensor:
        ...
        return

    def apply_calibration(self, preds, uncs) -> Tensor:
        ...
        return


@UncertaintyCalibratorRegistry.register("conformal-multiclass")
class ConformalMulticlassCalibrator(UncertaintyCalibrator):
    def calibrate(self, preds, uncs, targets, mask) -> Tensor:
        ...
        return

    def apply_calibration(self, preds, uncs) -> Tensor:
        ...
        return


@UncertaintyCalibratorRegistry.register("conformal-adaptive")
class ConformalAdaptiveMulticlassCalibrator(UncertaintyCalibrator):
    def calibrate(self, preds, uncs, targets, mask) -> Tensor:
        ...
        return

    def apply_calibration(self, preds, uncs) -> Tensor:
        ...
        return


@UncertaintyCalibratorRegistry.register("conformal-regression")
class ConformalRegressionCalibrator(UncertaintyCalibrator):
    def calibrate(self, preds, uncs, targets, mask) -> Tensor:
        ...
        return

    def apply_calibration(self, preds, uncs) -> Tensor:
        ...
        return


@UncertaintyCalibratorRegistry.register("conformal-quantile-regression")
class ConformalQuantileRegressionCalibrator(UncertaintyCalibrator):
    def calibrate(self, preds, uncs, targets, mask) -> Tensor:
        ...
        return

    def apply_calibration(self, preds, uncs) -> Tensor:
        ...
        return


@UncertaintyCalibratorRegistry.register("isotonic")
class IsotonicCalibrator(UncertaintyCalibrator):
    def calibrate(self, preds, uncs, targets, mask) -> Tensor:
        ...
        return

    def apply_calibration(self, preds, uncs) -> Tensor:
        ...
        return


@UncertaintyCalibratorRegistry.register("isotonic-multiclass")
class IsotonicMulticlassCalibrator(UncertaintyCalibrator):
    def calibrate(self, preds, uncs, targets, mask) -> Tensor:
        ...
        return

    def apply_calibration(self, preds, uncs) -> Tensor:
        ...
        return
