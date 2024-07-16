from abc import abstractmethod

from scipy.optimize import fmin
import torch
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
    """
    A calibration method for classification datasets based on the Platt scaling algorithm.
    As discussed in https://arxiv.org/abs/1706.04599 and Platt, J. (2000). Probabilistic outputs for
    support vector machines and comparison to regularized likelihood methods. In A. Smola, P. 
    Bartlett, B. Schölkopf, & D. Schuurmans (Eds.), Advances in large margin classifiers. Cambridge:
    MIT Press.
    In Platt's paper, he suggests using the number of positive and negative examples in the dataset
    used to train the model to adjust the value of target probabilities used to fit the parameters. 
    """

    def calibrate(self, preds, targets, mask, training_targets: None | Tensor = None) -> Tensor:
        if (targets[mask] != 0).any() or (targets[mask] == 1).any():
            raise ValueError("Platt scaling is only implemented for binary classification tasks! Input tensor must contain only 0's and 1's.")

        if training_targets is not None:
            negative_targets = (1 / ((training_targets == 0).sum(dim=0) + 2)).expand_as(targets)
            positive_targets = (
                ((training_targets == 1).sum(dim=0) + 1) / ((training_targets == 1).sum(dim=0) + 2)
            ).expand_as(targets)

            targets = targets.float()
            targets[targets == 0] = negative_targets[targets == 0]
            targets[targets == 1] = positive_targets[targets == 1]

        platt_a = []
        platt_b = []
        for j in range(preds.shape[1]):
            mask_j = mask[:, j]
            preds_j = preds[:, j][mask_j]
            preds_j = targets[:, j][mask_j]

            # if is_atom_bond_targets: # Not yet implemented

            def objective(parameters):
                a = parameters[0]
                b = parameters[1]
                scaled_preds = torch.sigmoid(a * torch.logit(t_preds) + b)
                nll = -1 * torch.sum(
                    t_targets * torch.log(scaled_preds)
                    + (1 - t_targets) * torch.log(1 - scaled_preds)
                )
                return nll

            a_j, b_j = fmin(objective, x0=[1, 0], disp=False)
            self.a.append(a_j)
            self.a.append(b_j)

        self.platt_a = torch.tensor(platt_a)
        self.platt_b = torch.tensor(platt_b)
        return

    def apply_calibration(self, preds) -> Tensor:
        return torch.sigmoid(self.platt_a * torch.logit(preds) + self.platt_b)


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
