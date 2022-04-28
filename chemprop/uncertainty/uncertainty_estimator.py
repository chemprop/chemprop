from typing import Iterator

from chemprop.data import MoleculeDataset, StandardScaler
from chemprop.models import MoleculeModel
from .uncertainty_calibrator import UncertaintyCalibrator
from .uncertainty_predictor import uncertainty_predictor_builder


class UncertaintyEstimator:
    def __init__(self, test_data: MoleculeDataset,
                       models: Iterator[MoleculeModel],
                       scalers: Iterator[StandardScaler],
                       uncertainty_method: str,
                       dataset_type: str,
                       loss_function: str,
                       batch_size: int,
                       num_workers: int,
                       dropout_sampling_size: int,
                       ):
        self.uncertainty_method = uncertainty_method

        self.predictor = uncertainty_predictor_builder(
            test_data=test_data,
            models=models,
            scalers=scalers,
            uncertainty_method=uncertainty_method,
            dataset_type=dataset_type,
            loss_function=loss_function,
            batch_size=batch_size,
            num_workers=num_workers,
            dropout_sampling_size=dropout_sampling_size,
        )
        self.label = self.predictor.label

    def calculate_uncertainty(self, calibrator: UncertaintyCalibrator = None):
        """
        Return values for the prediction and uncertainty metric. 
        If a calibrator is provided, returns a calibrated metric of the type specified.
        """

        if calibrator is not None:
            self.label = calibrator.label
            cal_preds, cal_unc = calibrator.apply_calibration(uncal_predictor=self.predictor)
            return cal_preds, cal_unc
        else:
            uncal_preds = self.predictor.get_uncal_preds()
            uncal_output = self.predictor.get_uncal_output()
            return uncal_preds, uncal_output