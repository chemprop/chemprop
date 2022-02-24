from typing import Iterator

import numpy as np

from chemprop.data import MoleculeDataset, StandardScaler
from chemprop.models import MoleculeModel
from .uncertainty_predictor import uncertainty_predictor_builder

class UncertaintyCalibrator:
    def __init__(self, uncertainty_method: str,
                       calibration_data: MoleculeDataset,
                       calibration_metric: str,
                       models: Iterator[MoleculeModel],
                       scalers: Iterator[StandardScaler],
                       dataset_type: str,
                       ):
        self.calibration_data = calibration_data
        self.uncertainty_method = uncertainty_method
        self.calibration_metric = calibration_metric
        self.models = models
        self.scalers = scalers
        self.dataset_type = dataset_type

        self.raise_argument_errors()

        self.calibration_data_estimator = uncertainty_predictor_builder(
            test_data=calibration_data,
            models=models,
            scalers=scalers,
            dataset_type=dataset_type,
            return_invalid_smiles=False,
        )
    
    def raise_argument_errors(self):
        """
        Raise errors for incompatibilities between dataset type and uncertainty method, or similar.
        """
        pass

    def calibrate(self, means, unc_parameters):
        """
        Take in predicted means and uncertainty parameters from a model
        """
        pass


class HistogramCalibrator(UncertaintyCalibrator):
    def __init__(self, uncertainty_method: str, calibration_data: MoleculeDataset, calibration_metric: str, models: Iterator[MoleculeModel], scalers: Iterator[StandardScaler]):
        super().__init__(uncertainty_method, calibration_data, calibration_metric, models, scalers)


def uncertainty_calibrator_builder(calibration_method: str,
                                   uncertainty_method: str,
                                   calibration_data: MoleculeDataset,
                                   calibration_metric: str,
                                   models: Iterator[MoleculeModel],
                                   scalers: Iterator[StandardScaler],
                                   dataset_type: str,
                                   ) -> UncertaintyCalibrator:
    """
    
    """
    supported_calibrators = {
        'histogram': HistogramCalibrator
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
        )
    return calibrator