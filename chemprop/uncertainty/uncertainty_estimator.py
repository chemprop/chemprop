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
                       return_invalid_smiles: bool = True,
                       return_pred_dict: bool = False,
                       ):
        self.test_data = test_data
        self.models = models
        self.scalers = scalers
        self.dataset_type = dataset_type
        self.return_invalid_smiles = return_invalid_smiles
        self.return_pred_dict = return_pred_dict

        self.predictor = uncertainty_predictor_builder(
            test_data=test_data,
            models=models,
            scalers=scalers,
            dataset_type=dataset_type,
            return_invalid_smiles=return_invalid_smiles,
            return_pred_dict=return_pred_dict,
        )

    def calculate_uncertainty(self, calibrator: UncertaintyCalibrator = None):
        """
        Return values for the prediction and uncertainty metric. 
        If a calibrator is provided, returns a calibrated metric of the type specified.
        """
        pass