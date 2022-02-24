from typing import Iterator

import numpy as np

from chemprop.data import MoleculeDataset, StandardScaler
from chemprop.models import MoleculeModel


class UncertaintyPredictor:
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
    
    def raise_argument_errors(self):
        """
        Raise errors for incompatible dataset types or uncertainty methods, etc.
        """
        pass

    def calculate_predictions(self):
        """
        Calculate the uncalibrated predictions and store them as class attributes
        """
    
    def means(self):
        """Return the predicted values for the test data."""
        pass

    def uncal_vars(self):
        """Return the uncalibrated variances for the test data"""
        pass

    def unc_parameters(self):
        """Return a tuple of uncertainty parameters for the prediction"""
        pass

    def uncal_prob_of_prediction(self,targets):
        """
        For a given set of targets, return the uncalibrated probability 
        for the calculated mean prediction
        """
        pass


class MVEPredictor(UncertaintyPredictor):
    def __init__(self, test_data: MoleculeDataset, models: Iterator[MoleculeModel], scalers: Iterator[StandardScaler]):
        super().__init__(test_data, models, scalers)


def uncertainty_predictor_builder(uncertainty_method: str,
                                  test_data: MoleculeDataset,
                                  models: Iterator[MoleculeModel],
                                  scalers: Iterator[StandardScaler],
                                  dataset_type: str,
                                  return_invalid_smiles: bool = True,
                                  return_pred_dict: bool = False,
                                  ) -> UncertaintyPredictor:
    """
    
    """
    supported_predictors = {
        'mve': MVEPredictor
    }

    estimator_class = supported_predictors.get(uncertainty_method, None)
    
    if estimator_class is None:
        raise NotImplementedError(f'Uncertainty estimator type {uncertainty_method} is not currently supported. Avalable options are: {supported_predictors.keys()}')
    else:
        estimator = estimator_class(
            test_data=test_data,
            models=models,
            scalers=scalers,
            dataset_type=dataset_type,
            return_invalid_smiles=return_invalid_smiles,
            return_pred_dict=return_pred_dict,
        )
    return estimator