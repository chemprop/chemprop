from typing import Iterator

import numpy as np

from chemprop.data import MoleculeDataset, StandardScaler
from chemprop.models import MoleculeModel
from .uncertainty_calibrator import UncertaintyCalibrator
from .uncertainty_predictor import uncertainty_predictor_builder


class UncertaintyEstimator:
    def __init__(self, test_data: MoleculeDataset,
                       models: Iterator[MoleculeModel],
                       scalers: Iterator[StandardScaler],
                       dataset_type: str,
                       loss_function: str,
                       ):

        self.predictor = uncertainty_predictor_builder(
            test_data=test_data,
            models=models,
            scalers=scalers,
            dataset_type=dataset_type,
            loss_function=loss_function,
        )

    def calculate_uncertainty(self, calibrator: UncertaintyCalibrator = None):
        """
        Return values for the prediction and uncertainty metric. 
        If a calibrator is provided, returns a calibrated metric of the type specified.
        """
        means = self.predictor.get_means()

        if calibrator is not None:
            unc_params = self.predictor.get_unc_parameters()
            cal_unc = calibrator.apply_calibration(means, unc_params)
            return means, cal_unc
        else:
            uncal_vars = self.predictor.get_uncal_vars()
            return means, uncal_vars