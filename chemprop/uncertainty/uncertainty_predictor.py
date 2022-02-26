from typing import Iterator, List

import numpy as np
from scipy.special import erf

from chemprop.data import MoleculeDataset, StandardScaler, MoleculeDataLoader
from chemprop.models import MoleculeModel
from chemprop.train import predict

# Should there be an intermediate class inheritor for dataset type?
class UncertaintyPredictor:
    def __init__(self, test_data: MoleculeDataset,
                       models: Iterator[MoleculeModel],
                       scalers: Iterator[StandardScaler],
                       dataset_type: str,
                       loss_function: str,
                       batch_size: int,
                       num_workers: int,
                       ):
        self.test_data = test_data
        self.models = models
        self.scalers = scalers
        self.dataset_type = dataset_type
        self.loss_function = loss_function
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.raise_argument_errors()
        self.test_data_loader=MoleculeDataLoader(
            dataset=self.test_data,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )
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
    
    def get_uncal_preds(self):
        """Return the predicted values for the test data."""
        return self.uncal_preds

    def get_uncal_vars(self):
        """Return the uncalibrated variances for the test data"""
        return self.uncal_vars

    def get_unc_parameters(self):
        """Return a tuple of uncertainty parameters for the prediction"""
        return self.unc_parameters

    def uncal_prob_of_prediction(self,targets: List[List[float]]):
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
                                  batch_size: int,
                                  num_workers: int,
                                  ) -> UncertaintyPredictor:
    """
    
    """
    supported_predictors = {
        'mve': MVEPredictor,
        'ensemble': EnsemblePredictor,
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
            batch_size=batch_size,
            num_workers=num_workers,
        )
    return estimator