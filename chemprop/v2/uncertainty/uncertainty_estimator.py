from typing import Iterator, List

from chemprop.data import MoleculeDataset, StandardScaler
from chemprop.data.data import MoleculeDataLoader
from chemprop.models import MoleculeModel
from chemprop.uncertainty.uncertainty_calibrator import UncertaintyCalibrator
from chemprop.uncertainty.uncertainty_predictor import build_uncertainty_predictor


class UncertaintyEstimator:
    def __init__(
        self,
        test_data: MoleculeDataset,
        test_data_loader: MoleculeDataLoader,
        models: Iterator[MoleculeModel],
        scalers: Iterator[StandardScaler],
        num_models: int,
        dataset_type: str,
        individual_ensemble_predictions: bool,
        spectra_phase_mask: List[List[bool]],
    ):
        self.predictor = build_uncertainty_predictor(
            test_data=test_data,
            test_data_loader=test_data_loader,
            models=models,
            scalers=scalers,
            num_models=num_models,
            dataset_type=dataset_type,
            individual_ensemble_predictions=individual_ensemble_predictions,
            spectra_phase_mask=spectra_phase_mask,
        )
        self.label = self.predictor.label

    def calculate_uncertainty(self, calibrator: UncertaintyCalibrator = None):
        """
        Return values for the prediction and uncertainty metric.
        If a calibrator is provided, returns a calibrated metric of the type specified.
        """

        if calibrator is not None:
            self.label = calibrator.label
            cal_preds, cal_unc = calibrator.apply_calibration(
                uncal_predictor=self.predictor
            )
            return cal_preds, cal_unc
        else:
            uncal_preds = self.predictor.get_uncal_preds()
            uncal_output = self.predictor.get_uncal_output()
            return uncal_preds, uncal_output

    def individual_predictions(self):
        """
        Return separate predictions made by each individual model in an ensemble of models.
        """
        return self.predictor.get_individual_preds()
