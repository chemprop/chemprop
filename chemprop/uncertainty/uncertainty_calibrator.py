from typing import Iterator

import numpy as np
from scipy.special import erfinv
from scipy.optimize import root

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
    def __init__(self, uncertainty_method: str,
                       calibration_data: MoleculeDataset,
                       calibration_metric: str,
                       models: Iterator[MoleculeModel],
                       scalers: Iterator[StandardScaler],
                       dataset_type: str,
                       loss_function: str,
                       batch_size: int,
                       num_workers: int,
                       ):
        self.calibration_data = calibration_data
        self.uncertainty_method = uncertainty_method
        self.calibration_metric = calibration_metric
        self.models = models
        self.scalers = scalers
        self.dataset_type = dataset_type
        self.loss_function = loss_function
        self.batch_size = batch_size
        self.num_workers = num_workers

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
        calibration_functions = {
            'stdev': (self.calibrate_stdev, self.apply_stdev),
            '95interval': (self.calibrate_95interval, self.apply_95interval),
            'confidence': (self.calibrate_confidence, self.apply_confidence),
        }[self.calibration_metric]
        calibration_functions[0]()
        self._apply_calibration = calibration_functions[1]

    def apply_calibration(self, uncal_preds, uncal_vars, unc_parameters, uncertainty_method):
        """
        Take in predictions and uncertainty parameters from a model and apply the calibration method using fitted parameters.
        """
        return self._apply_calibration(uncal_preds, uncal_vars, unc_parameters, uncertainty_method)

    def calibrate_stdev(self):
        """
        Fit calibration method for stdev metric.
        """
        raise NotImplementedError(f'The calibration metric stdev is not implemented with the chosen calibration method.')

    def calibrate_95interval(self):
        """
        Fit calibration method for 95interval metric.
        """
        raise NotImplementedError(f'The calibration metric 95interval is not implemented with the chosen calibration method.')

    def calibrate_confidence(self):
        """
        Fit calibration method for confidence metric.
        """
        raise NotImplementedError(f'The calibration metric confidence is not implemented with the chosen calibration method.')

    def apply_stdev(self, uncal_preds, uncal_vars, unc_parameters, uncertainty_method):
        """
        Take in predications and uncertainty parameters from a model and return a calibrated standard deviation uncertainty.
        """
        raise NotImplementedError(f'The calibration metric stdev is not implemented with the chosen calibration method.')

    def apply_95interval(self, uncal_preds, uncal_vars, unc_parameters, uncertainty_method):
        """
        Take in predications and uncertainty parameters from a model and return a calibrated standard deviation uncertainty.
        """
        raise NotImplementedError(f'The calibration metric 95interval is not implemented with the chosen calibration method.')

    def apply_confidence(self, uncal_preds, uncal_vars, unc_parameters, uncertainty_method):
        """
        Take in predications and uncertainty parameters from a model and return a calibrated standard deviation uncertainty.
        """
        raise NotImplementedError(f'The calibration metric confidence is not implemented with the chosen calibration method.')


class ZscoreCalibrator(UncertaintyCalibrator):
    def __init__(self, uncertainty_method: str, calibration_data: MoleculeDataset, calibration_metric: str, models: Iterator[MoleculeModel], scalers: Iterator[StandardScaler], dataset_type: str, loss_function: str, batch_size: int, num_workers: int):
        super().__init__(uncertainty_method, calibration_data, calibration_metric, models, scalers, dataset_type, loss_function, batch_size, num_workers)
        self.raise_argument_errors()

    def raise_argument_errors(self):
        super().raise_argument_errors()
        if self.dataset_type != 'regression':
            raise ValueError('Z Score Calibration is only compatible with regression datasets.')

    def calibrate_stdev(self):
        uncal_preds = np.array(self.calibration_predictor.get_uncal_preds()) # shape(data, tasks)
        uncal_vars = np.array(self.calibration_predictor.get_uncal_vars())
        targets = np.array(self.calibration_data.targets())
        zscore_preds = (uncal_preds - targets) / np.sqrt(uncal_vars)
        abs_zscore_preds = np.abs(zscore_preds)

        def objective(scaler_values: np.ndarray):
            cal_z = abs_zscore_preds / scaler_values
            return calibration_normal_auc(cal_z)

        initial_guess = np.std(zscore_preds, axis=0, keepdims=True)
        sol = root(objective, initial_guess)
        self.stdev_scaling = sol.x

    def calibrate_95interval(self):
        uncal_preds = np.array(self.calibration_predictor.get_uncal_preds()) # shape(data, tasks)
        uncal_vars = np.array(self.calibration_predictor.get_uncal_vars())
        targets = np.array(self.calibration_data.targets())
        zscore_preds = np.abs(uncal_preds - targets) / np.sqrt(uncal_vars)
        self.stdev_scaling = np.percentile(zscore_preds, 95, axis=0, keepdims=True)

    def apply_stdev(self, uncal_preds, uncal_vars, unc_parameters, uncertainty_method):
        cal_stdev = np.sqrt(uncal_vars) * self.stdev_scaling
        return uncal_preds, cal_stdev.tolist()

    def apply_95interval(self, uncal_preds, uncal_vars, unc_parameters, uncertainty_method):
        cal_stdev = np.sqrt(uncal_vars) * self.stdev_scaling
        return uncal_preds, cal_stdev.tolist()


def uncertainty_calibrator_builder(calibration_method: str,
                                   uncertainty_method: str,
                                   calibration_data: MoleculeDataset,
                                   calibration_metric: str,
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
        'zscorefit': ZscoreCalibrator,
    }

    calibrator_class = supported_calibrators.get(calibration_method, None)
    
    if calibrator_class is None:
        raise NotImplementedError(f'Calibrator type {calibration_method} is not currently supported. Avalable options are: {supported_calibrators.keys()}')
    else:
        calibrator = calibrator_class(
            uncertainty_method=uncertainty_method,
            calibration_data=calibration_data,
            calibration_metric=calibration_metric,
            models=models,
            scalers=scalers,
            dataset_type=dataset_type,
            loss_function=loss_function,
            batch_size=batch_size,
            num_workers=num_workers,
        )
    return calibrator