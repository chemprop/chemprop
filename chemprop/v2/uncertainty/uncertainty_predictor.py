from abc import ABC, abstractmethod
from typing import Iterator, List

import numpy as np
from tqdm import tqdm

from chemprop.v2.data import MoleculeDataset, MolGraphDataLoader
from sklearn.preprocessing import StandardScaler
from chemprop.v2.models import MPNN, ClassificationMPNN, DirichletClassificationMPNN, MulticlassMPNN, DirichletMulticlassMPNN, RegressionMPNN, MveRegressionMPNN, SpectralMPNN 
from chemprop.v2.train.predict import predict
from chemprop.spectra_utils import normalize_spectra, roundrobin_sid


class UncertaintyPredictor(ABC):
    """
    A class for making model predictions and associated predictions of
    prediction uncertainty according to the chosen uncertainty method.
    """
    def __init__(
        self,
        test_data: MoleculeDataset,
        test_data_loader: MolGraphDataLoader,
        models: Iterator[Union[ClassificationMPNN, DirichletClassificationMPNN, MulticlassMPNN, DirichletMulticlassMPNN, RegressionMPNN, MveRegressionMPNN, SpectralMPNN]],
        dataset_type: str,
        individual_ensemble_predictions: bool = False,
        spectra_phase_mask: List[List[bool]] = None,
    ):
        self.test_data = test_data
        self.models = models
        self.dataset_type = dataset_type
        self.uncal_preds = None
        self.uncal_vars = None
        self.num_models = num_models
        self.individual_ensemble_predictions = individual_ensemble_predictions
        self.spectra_phase_mask = spectra_phase_mask

        self.raise_argument_errors()
        self.test_data_loader = test_data_loader
        self.calculate_predictions()

    @property
    @abstractmethod
    def label(self):
        """
        The string in saved results indicating the uncertainty method used.
        """

    def raise_argument_errors(self):
        """
        Raise errors for incompatible dataset types or uncertainty methods, etc.
        """

    @abstractmethod
    def calculate_predictions(self):
        """
        Calculate the uncalibrated predictions and store them as attributes
        """

    def get_uncal_preds(self):
        """
        Return the predicted values for the test data.
        """
        return self.uncal_preds

    def get_uncal_vars(self):
        """
        Return the uncalibrated variances for the test data
        """
        return self.uncal_vars

    @abstractmethod
    def get_uncal_output(self):
        """
        Return the uncalibrated uncertainty outputs for the test data
        """


class NoUncertaintyPredictor(UncertaintyPredictor):
    """
    Class that is used for predictions when no uncertainty method is selected.
    Model value predictions are made as normal but uncertainty output only returns "nan".
    """
    @property
    def label(self):
        return "no_uncertainty_method"

    def calculate_predictions(self):
        # for i, (model, scaler_list) in enumerate(
        #     tqdm(zip(self.models, self.scalers), total=self.num_models)
        # ):
        #     (
        #         scaler,
        #         features_scaler,
        #         atom_descriptor_scaler,
        #         bond_feature_scaler,
        #     ) = scaler_list
        #     if (
        #         features_scaler is not None
        #         or atom_descriptor_scaler is not None
        #         or bond_feature_scaler is not None
        #     ):
        #         self.test_data.reset_features_and_targets()
        #         if features_scaler is not None:
        #             self.test_data.normalize_features(features_scaler)
        #         if atom_descriptor_scaler is not None:
        #             self.test_data.normalize_features(
        #                 atom_descriptor_scaler, scale_atom_descriptors=True
        #             )
        #         if bond_feature_scaler is not None:
        #             self.test_data.normalize_features(
        #                 bond_feature_scaler, scale_bond_features=True
        #             )
        for i, model in enumerate(tqdm(self.models)):
            preds, _ = predict(
                model=model,
                data_loader=self.test_data_loader,
            )
            if self.dataset_type == "spectra":
                preds = normalize_spectra(
                    spectra=preds,
                    phase_features=self.test_data.phase_features(),
                    phase_mask=self.spectra_phase_mask,
                    excluded_sub_value=float("nan"),
                )
            if i == 0:
                sum_preds = np.array(preds)
                if self.individual_ensemble_predictions:
                    individual_preds = np.expand_dims(np.array(preds), axis=-1)
            else:
                sum_preds += np.array(preds)
                if self.individual_ensemble_predictions:
                    individual_preds = np.append(individual_preds, np.expand_dims(preds, axis=-1), axis=-1)

        self.uncal_preds = (sum_preds / self.num_models).tolist()
        uncal_vars = np.zeros_like(sum_preds)
        uncal_vars[:] = np.nan
        self.uncal_vars = uncal_vars
        if self.individual_ensemble_predictions:
            self.individual_preds = individual_preds.tolist()
        
    def get_uncal_output(self):
        return self.uncal_vars

def build_uncertainty_predictor(
    test_data: MoleculeDataset,
    test_data_loader: MolGraphDataLoader,
    models: Iterator[Union[ClassificationMPNN, DirichletClassificationMPNN, MulticlassMPNN, DirichletMulticlassMPNN, RegressionMPNN, MveRegressionMPNN, SpectralMPNN]],
    dataset_type: str,
    individual_ensemble_predictions: bool,
    spectra_phase_mask: List[List[bool]],
) -> UncertaintyPredictor:
  
    predictor = NoUncertaintyPredictor(
        test_data=test_data,
        test_data_loader=test_data_loader,
        models=models,
        dataset_type=dataset_type,
        individual_ensemble_predictions=individual_ensemble_predictions,
        spectra_phase_mask=spectra_phase_mask,
    )
    return predictor
