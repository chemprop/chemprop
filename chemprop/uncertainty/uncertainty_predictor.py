from typing import Iterator

import numpy as np

from chemprop.data import MoleculeDataset, StandardScaler
from chemprop.models import MoleculeModel


class UncertaintyPredictor:
    def __init__(self, test_data: MoleculeDataset,
                       models: Iterator[MoleculeModel],
                       scalers: Iterator[StandardScaler],
                       dataset_type: str,
                       loss_function: str,
                       ):
        self.test_data = test_data
        self.models = models
        self.scalers = scalers
        self.dataset_type = dataset_type
        self.loss_function = loss_function

        self.raise_argument_errors()
        self.calculate_predictions()
    
    def raise_argument_errors(self):
        """
        Raise errors for incompatible dataset types or uncertainty methods, etc.
        """
        pass

    def calculate_predictions(self):
        """
        Calculate the uncalibrated predictions and store them as attributes
        """
        pass
    
    def get_means(self):
        """Return the predicted values for the test data."""
        return self.means

    def get_uncal_vars(self):
        """Return the uncalibrated variances for the test data"""
        return self.uncal_vars

    def get_unc_parameters(self):
        """Return a tuple of uncertainty parameters for the prediction"""
        return self.unc_parameters

    def uncal_prob_of_prediction(self,targets):
        """
        For a given set of targets, return the uncalibrated probability 
        for the calculated mean prediction
        """
        pass


class MVEPredictor(UncertaintyPredictor):
    def __init__(self, test_data: MoleculeDataset, models: Iterator[MoleculeModel], scalers: Iterator[StandardScaler], dataset_type: str, loss_function: str):
        super().__init__(test_data, models, scalers, dataset_type, loss_function)

    def raise_argument_errors(self):
        if self.loss_function != 'mve':
            raise ValueError('In order to use mve uncertainty, trained models must have used mve loss function.')

    def uncal_vars(self):
        "do a thing"


def uncertainty_predictor_builder(uncertainty_method: str,
                                  test_data: MoleculeDataset,
                                  models: Iterator[MoleculeModel],
                                  scalers: Iterator[StandardScaler],
                                  dataset_type: str,
                                  loss_function: str,
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
            loss_function=loss_function,
        )
    return estimator