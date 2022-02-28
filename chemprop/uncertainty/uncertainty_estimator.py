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
        )

    def calculate_uncertainty(self, calibrator: UncertaintyCalibrator = None):
        """
        Return values for the prediction and uncertainty metric. 
        If a calibrator is provided, returns a calibrated metric of the type specified.
        """
        uncal_preds = self.predictor.get_uncal_preds()
        uncal_vars = self.predictor.get_uncal_vars()

        if calibrator is not None:
            unc_params = self.predictor.get_unc_parameters()
            cal_preds, cal_unc = calibrator.apply_calibration(uncal_preds=uncal_preds, uncal_vars=uncal_vars, unc_parameters=unc_params, uncertainty_method=self.uncertainty_method)
            return cal_preds, cal_unc
        else:
            return uncal_preds, uncal_vars