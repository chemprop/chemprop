from typing import Iterator

import numpy as np
from scipy.special import erfinv, erf
from scipy.optimize import root, fmin

from chemprop.data import MoleculeDataset, StandardScaler
from chemprop.models import MoleculeModel
from .uncertainty_predictor import uncertainty_predictor_builder
from .utils import calibration_normal_auc

class UncertaintyCalibrator:
    """
    Uncertainty calibrator class. Subclasses for each uncertainty calibration 
    method. Subclasses should override the calibrate and apply functions for 
    implemented metrics.
    """
    def __init__(
        self,
        uncertainty_method: str,
        interval_percentile: int,
        regression_calibrator_metric: str,
        calibration_data: MoleculeDataset,
        models: Iterator[MoleculeModel],
        scalers: Iterator[StandardScaler],
        dataset_type: str,
        loss_function: str,
        batch_size: int,
        num_workers: int,
    ):
        self.calibration_data = calibration_data
        self.regression_calibrator_metric = regression_calibrator_metric
        self.interval_percentile = interval_percentile

        self.raise_argument_errors()

        self.calibration_predictor = uncertainty_predictor_builder(
            test_data=calibration_data,
            models=models,
            scalers=scalers,
            dataset_type=dataset_type,
            batch_size=batch_size,
            num_workers=num_workers,
            loss_function=loss_function,
            uncertainty_method=uncertainty_method,
        )

        self.calibrate()
    
    def raise_argument_errors(self):
        """
        Raise errors for incompatibilities between dataset type and uncertainty method, or similar.
        """
        pass

    def calibrate(self):
        """
        Fit calibration method for the calibration data.
        """
        pass

    def apply_calibration(self, uncal_preds, uncal_vars, uncal_confidence, uncertainty_method):
        """
        Take in predictions and uncertainty parameters from a model and apply the calibration method using fitted parameters.
        """
        pass


class ZScalingCalibrator(UncertaintyCalibrator):
    def __init__(
        self,
        uncertainty_method: str,
        interval_percentile: int,
        regression_calibrator_metric: str,
        calibration_data: MoleculeDataset,
        models: Iterator[MoleculeModel],
        scalers: Iterator[StandardScaler],
        dataset_type: str,
        loss_function: str,
        batch_size: int,
        num_workers: int,
    ):
        super().__init__(
            uncertainty_method=uncertainty_method,
            interval_percentile=interval_percentile,
            regression_calibrator_metric=regression_calibrator_metric,
            calibration_data=calibration_data,
            models=models,
            scalers=scalers,
            dataset_type=dataset_type,
            loss_function=loss_function,
            batch_size=batch_size,
            num_workers=num_workers,
        )
        if self.regression_calibrator_metric == 'stdev':
            self.label = f'{uncertainty_method}_zscaling_stdev'
        else: # interval
            self.label = f'{uncertainty_method}_zscaling_{interval_percentile}interval'

    def raise_argument_errors(self):
        super().raise_argument_errors()
        if self.dataset_type != 'regression':
            raise ValueError('Z Score Scaling is only compatible with regression datasets.')

    def calibrate(self):
        uncal_preds = np.array(self.calibration_predictor.get_uncal_preds()) # shape(data, tasks)
        uncal_vars = np.array(self.calibration_predictor.get_uncal_vars())
        targets = np.array(self.calibration_data.targets())
        errors = uncal_preds - targets
        zscore_preds = errors / np.sqrt(uncal_vars)

        def objective(scaler_values: np.ndarray):
            scaled_vars = uncal_vars * scaler_values ** 2
            nll = np.log(2 * np.pi * scaled_vars) / 2 + (errors) ** 2 / (2 * scaled_vars)
            nll = np.sum(nll, axis=0)
            return nll

        initial_guess = np.std(zscore_preds, axis=0, keepdims=True)
        sol = fmin(objective, initial_guess)
        stdev_scaling = sol[0]
        if self.regression_calibrator_metric == 'stdev':
            self.scaling = stdev_scaling
        else: # interval
            interval_scaling = stdev_scaling * erfinv(self.interval_percentile/100) * np.sqrt(2)
            self.scaling = interval_scaling

    def apply_calibration(self, uncal_preds, uncal_vars, uncal_confidence, uncertainty_method):
        cal_stdev = np.sqrt(uncal_vars) * self.scaling
        return uncal_preds, cal_stdev.tolist()


class ZCrudeCalibrator(UncertaintyCalibrator):
    def __init__(
        self,
        uncertainty_method: str,
        interval_percentile: int,
        regression_calibrator_metric: str,
        calibration_data: MoleculeDataset,
        models: Iterator[MoleculeModel],
        scalers: Iterator[StandardScaler],
        dataset_type: str,
        loss_function: str,
        batch_size: int,
        num_workers: int,
    ):
        super().__init__(
            uncertainty_method=uncertainty_method,
            interval_percentile=interval_percentile,
            regression_calibrator_metric=regression_calibrator_metric,
            calibration_data=calibration_data,
            models=models,
            scalers=scalers,
            dataset_type=dataset_type,
            loss_function=loss_function,
            batch_size=batch_size,
            num_workers=num_workers,
        )
        self.label = f'{uncertainty_method}_zcrude_{interval_percentile}interval'

    def raise_argument_errors(self):
        super().raise_argument_errors()
        if self.dataset_type != 'regression':
            raise ValueError('Z Score Scaling is only compatible with regression datasets.')

    def calibrate(self):
        uncal_preds = np.array(self.calibration_predictor.get_uncal_preds()) # shape(data, tasks)
        uncal_vars = np.array(self.calibration_predictor.get_uncal_vars())
        targets = np.array(self.calibration_data.targets())
        abs_zscore_preds = np.abs(uncal_preds - targets) / np.sqrt(uncal_vars)
        interval_scaling = np.percentile(abs_zscore_preds, self.interval_percentile, axis=0, keepdims=True)
        self.scaling = interval_scaling

    def apply_calibration(self, uncal_preds, uncal_vars, uncal_confidence, uncertainty_method):
        cal_stdev = np.sqrt(uncal_vars) * self.scaling
        return uncal_preds, cal_stdev.tolist()


class TScalingCalibrator(UncertaintyCalibrator):
    def __init__(
        self,
        uncertainty_method: str,
        interval_percentile: int,
        regression_calibrator_metric: str,
        calibration_data: MoleculeDataset,
        models: Iterator[MoleculeModel],
        scalers: Iterator[StandardScaler],
        dataset_type: str,
        loss_function: str,
        batch_size: int,
        num_workers: int,
    ):
        super().__init__(
            uncertainty_method=uncertainty_method,
            interval_percentile=interval_percentile,
            regression_calibrator_metric=regression_calibrator_metric,
            calibration_data=calibration_data,
            models=models,
            scalers=scalers,
            dataset_type=dataset_type,
            loss_function=loss_function,
            batch_size=batch_size,
            num_workers=num_workers,
        )
        if self.regression_calibrator_metric == 'stdev':
            self.label = f'{uncertainty_method}_tscaling_stdev'
        else: # interval
            self.label = f'{uncertainty_method}_tscaling_{interval_percentile}interval'

    def raise_argument_errors(self):
        super().raise_argument_errors()
        if self.dataset_type != 'regression':
            raise ValueError('T Score Scaling is only compatible with regression datasets.')

    def calibrate(self):
        uncal_preds = np.array(self.calibration_predictor.get_uncal_preds()) # shape(data, tasks)
        uncal_vars = np.array(self.calibration_predictor.get_uncal_vars())
        targets = np.array(self.calibration_data.targets())
        errors = uncal_preds - targets
        zscore_preds = errors / np.sqrt(uncal_vars)

        def objective(scaler_values: np.ndarray):
            scaled_vars = uncal_vars * scaler_values ** 2
            nll = np.log(2 * np.pi * scaled_vars) / 2 + (errors) ** 2 / (2 * scaled_vars)
            nll = np.sum(nll, axis=0)
            return nll

        initial_guess = np.std(zscore_preds, axis=0, keepdims=True)
        sol = fmin(objective, initial_guess)
        stdev_scaling = sol[0]
        if self.interval_percentile is None: # stdev metric
            self.scaling = stdev_scaling
        else:
            interval_scaling = stdev_scaling * erfinv(self.interval_percentile/100) * np.sqrt(2)
            self.scaling = interval_scaling

    def apply_calibration(self, uncal_preds, uncal_vars, uncal_confidence, uncertainty_method):
        cal_stdev = np.sqrt(uncal_vars) * self.scaling
        return uncal_preds, cal_stdev.tolist()


class TCrudeCalibrator(UncertaintyCalibrator):
    def __init__(
        self,
        uncertainty_method: str,
        interval_percentile: int,
        regression_calibrator_metric: str,
        calibration_data: MoleculeDataset,
        models: Iterator[MoleculeModel],
        scalers: Iterator[StandardScaler],
        dataset_type: str,
        loss_function: str,
        batch_size: int,
        num_workers: int,
    ):
        super().__init__(
            uncertainty_method=uncertainty_method,
            interval_percentile=interval_percentile,
            regression_calibrator_metric=regression_calibrator_metric,
            calibration_data=calibration_data,
            models=models,
            scalers=scalers,
            dataset_type=dataset_type,
            loss_function=loss_function,
            batch_size=batch_size,
            num_workers=num_workers,
        )
        self.label = f'{uncertainty_method}_tcrude_{interval_percentile}interval'

    def raise_argument_errors(self):
        super().raise_argument_errors()
        if self.dataset_type != 'regression':
            raise ValueError('T Score Scaling is only compatible with regression datasets.')

    def calibrate(self):
        uncal_preds = np.array(self.calibration_predictor.get_uncal_preds()) # shape(data, tasks)
        uncal_vars = np.array(self.calibration_predictor.get_uncal_vars())
        targets = np.array(self.calibration_data.targets())
        abs_zscore_preds = np.abs(uncal_preds - targets) / np.sqrt(uncal_vars)
        interval_scaling = np.percentile(abs_zscore_preds, self.interval_percentile, axis=0, keepdims=True)
        self.scaling = interval_scaling

    def apply_calibration(self, uncal_preds, uncal_vars, uncal_confidence, uncertainty_method):
        cal_stdev = np.sqrt(uncal_vars) * self.scaling
        return uncal_preds, cal_stdev.tolist()


def uncertainty_calibrator_builder(
    calibration_method: str,
    uncertainty_method: str,
    regression_calibrator_metric: str,
    interval_percentile: int,
    calibration_data: MoleculeDataset,
    models: Iterator[MoleculeModel],
    scalers: Iterator[StandardScaler],
    dataset_type: str,
    loss_function: str,
    batch_size: int,
    num_workers: int,
    ) -> UncertaintyCalibrator:
    """
    
    """
    supported_calibrators = {
        'zscaling': ZScalingCalibrator,
        'zcrude': ZCrudeCalibrator,
    }

    calibrator_class = supported_calibrators.get(calibration_method, None)
    
    if calibrator_class is None:
        raise NotImplementedError(f'Calibrator type {calibration_method} is not currently supported. Avalable options are: {supported_calibrators.keys()}')
    else:
        calibrator = calibrator_class(
            uncertainty_method=uncertainty_method,
            regression_calibrator_metric=regression_calibrator_metric,
            interval_percentile=interval_percentile,
            calibration_data=calibration_data,
            models=models,
            scalers=scalers,
            dataset_type=dataset_type,
            loss_function=loss_function,
            batch_size=batch_size,
            num_workers=num_workers,
        )
    return calibrator