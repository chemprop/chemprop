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
        self.uncal_preds = None
        self.uncal_vars = None
        self.unc_parameters = None

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


class MVEPredictor(UncertaintyPredictor):
    def __init__(self, test_data: MoleculeDataset, models: Iterator[MoleculeModel], scalers: Iterator[StandardScaler], dataset_type: str, loss_function: str, batch_size: int, num_workers: int):
        super().__init__(test_data, models, scalers, dataset_type, loss_function, batch_size, num_workers)

    def raise_argument_errors(self):
        super().raise_argument_errors()
        if self.loss_function != 'mve':
            raise ValueError('In order to use mve uncertainty, trained models must have used mve loss function.')

    def calculate_predictions(self):
        num_models = len(self.models)
        for i in range(num_models):

            scaler, features_scaler, atom_descriptor_scaler, bond_feature_scaler = self.scalers[i]
            if features_scaler is not None or atom_descriptor_scaler is not None or bond_feature_scaler is not None:
                self.test_data.reset_features_and_targets()
                if features_scaler is not None:
                    self.test_data.normalize_features(features_scaler)
                if atom_descriptor_scaler is not None:
                    self.test_data.normalize_features(atom_descriptor_scaler, scale_atom_descriptors=True)
                if bond_feature_scaler is not None:
                    self.test_data.normalize_features(bond_feature_scaler, scale_bond_features=True)

            preds, var = predict(
                model=self.models[i],
                data_loader=self.test_data_loader,
                scaler=scaler,
                return_unc_parameters=True,
            )
            if i == 0:
                sum_preds = np.array(preds)
                sum_squared = np.square(preds)
                sum_vars = np.array(var)
            else:
                sum_preds += np.array(preds)
                sum_squared += np.square(preds)
                sum_vars += np.array(var)

        uncal_preds = sum_preds / num_models
        uncal_vars = (sum_vars + sum_squared) / num_models - np.square(sum_preds / num_models)
        uncal_preds, uncal_vars = uncal_preds.tolist(), uncal_vars.tolist()
        self.unc_parameters = self.uncal_vars


class EnsemblePredictor(UncertaintyPredictor):
    def __init__(self, test_data: MoleculeDataset, models: Iterator[MoleculeModel], scalers: Iterator[StandardScaler], dataset_type: str, loss_function: str, batch_size: int, num_workers: int):
        super().__init__(test_data, models, scalers, dataset_type, loss_function, batch_size, num_workers)

    def raise_argument_errors(self):
        super().raise_argument_errors()
        if len(self.models) == 1:
            raise ValueError('Ensemble method for uncertainty is only available when multiple models are provided.')
    
    def calculate_predictions(self):
        num_models = len(self.models)
        for i in range(num_models):
            scaler, features_scaler, atom_descriptor_scaler, bond_feature_scaler = self.scalers[i]
            if features_scaler is not None or atom_descriptor_scaler is not None or bond_feature_scaler is not None:
                self.test_data.reset_features_and_targets()
                if features_scaler is not None:
                    self.test_data.normalize_features(features_scaler)
                if atom_descriptor_scaler is not None:
                    self.test_data.normalize_features(atom_descriptor_scaler, scale_atom_descriptors=True)
                if bond_feature_scaler is not None:
                    self.test_data.normalize_features(bond_feature_scaler, scale_bond_features=True)
            preds = predict(
                model=self.models[i],
                data_loader=self.test_data_loader,
                scaler=scaler,
                return_unc_parameters=False,
            )
            if i == 0:
                sum_preds = np.array(preds)
                sum_squared = np.square(preds)
            else:
                sum_preds += np.array(preds)
                sum_squared += np.square(preds)
        self.uncal_preds = sum_preds / num_models
        self.uncal_vars = sum_squared / num_models - np.square(sum_preds) / num_models ** 2
        self.unc_parameters = self.uncal_vars


class SigmoidPredictor(UncertaintyPredictor):
    """
    Class uses the [0,1] range of results from classification or multiclass models as the indicator of confidence.
    """
    def __init__(self, test_data: MoleculeDataset, models: Iterator[MoleculeModel], scalers: Iterator[StandardScaler], dataset_type: str, loss_function: str, batch_size: int, num_workers: int):
        super().__init__(test_data, models, scalers, dataset_type, loss_function, batch_size, num_workers)

    def raise_argument_errors(self):
        super().raise_argument_errors()
        if self.dataset_type not in ['classification', 'multiclass']:
            raise ValueError('Sigmoid uncertainty method must be used with dataset types classification or multiclass.')
    
    def calculate_predictions(self):
        num_models = len(self.models)
        for i in range(num_models):
            scaler, features_scaler, atom_descriptor_scaler, bond_feature_scaler = self.scalers[i]
            if features_scaler is not None or atom_descriptor_scaler is not None or bond_feature_scaler is not None:
                self.test_data.reset_features_and_targets()
                if features_scaler is not None:
                    self.test_data.normalize_features(features_scaler)
                if atom_descriptor_scaler is not None:
                    self.test_data.normalize_features(atom_descriptor_scaler, scale_atom_descriptors=True)
                if bond_feature_scaler is not None:
                    self.test_data.normalize_features(bond_feature_scaler, scale_bond_features=True)
            preds = predict(
                model=self.models[i],
                data_loader=self.test_data_loader,
                scaler=scaler,
                return_unc_parameters=False,
            )
            if i == 0:
                sum_preds = np.array(preds)
            else:
                sum_preds += np.array(preds)
        self.uncal_preds = sum_preds / num_models
        self.unc_parameters = self.uncal_preds


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
        'sigmoid': SigmoidPredictor,
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