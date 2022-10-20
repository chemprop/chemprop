from abc import ABC, abstractmethod
from typing import Iterator, List

import numpy as np
from tqdm import tqdm

from chemprop.data import MoleculeDataset, StandardScaler, MoleculeDataLoader
from chemprop.models import MoleculeModel
from chemprop.train.predict import predict
from chemprop.spectra_utils import normalize_spectra, roundrobin_sid


class UncertaintyPredictor(ABC):
    """
    A class for making model predictions and associated predictions of
    prediction uncertainty according to the chosen uncertainty method.
    """

    def __init__(
        self,
        test_data: MoleculeDataset,
        test_data_loader: MoleculeDataLoader,
        models: Iterator[MoleculeModel],
        scalers: Iterator[StandardScaler],
        num_models: int,
        dataset_type: str,
        loss_function: str,
        uncertainty_dropout_p: float,
        conformal_alpha: float,
        dropout_sampling_size: int,
        individual_ensemble_predictions: bool = False,
        spectra_phase_mask: List[List[bool]] = None,
    ):
        self.test_data = test_data
        self.models = models
        self.scalers = scalers
        self.dataset_type = dataset_type
        self.loss_function = loss_function
        self.uncal_preds = None
        self.uncal_vars = None
        self.uncal_intervals = None
        self.uncal_confidence = None
        self.individual_vars = None
        self.num_models = num_models
        self.uncertainty_dropout_p = uncertainty_dropout_p
        self.conformal_alpha = conformal_alpha
        self.dropout_sampling_size = dropout_sampling_size
        self.individual_ensemble_predictions = individual_ensemble_predictions
        self.spectra_phase_mask = spectra_phase_mask
        self.train_class_sizes = None

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

    def get_uncal_confidence(self):
        """
        Return the uncalibrated confidences for the test data
        """
        return self.uncal_confidence

    def get_individual_vars(self):
        """
        Return the variances predicted by each individual model in an ensemble.
        """
        return self.individual_vars

    def get_individual_preds(self):
        """
        Return the value predicted by each individual model in an ensemble.
        """
        return self.individual_preds

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
        for i, (model, scaler_list) in enumerate(
            tqdm(zip(self.models, self.scalers), total=self.num_models)
        ):
            (scaler, features_scaler, atom_descriptor_scaler, bond_feature_scaler) = scaler_list
            if (
                features_scaler is not None
                or atom_descriptor_scaler is not None
                or bond_feature_scaler is not None
            ):
                self.test_data.reset_features_and_targets()
                if features_scaler is not None:
                    self.test_data.normalize_features(features_scaler)
                if atom_descriptor_scaler is not None:
                    self.test_data.normalize_features(
                        atom_descriptor_scaler, scale_atom_descriptors=True
                    )
                if bond_feature_scaler is not None:
                    self.test_data.normalize_features(bond_feature_scaler, scale_bond_features=True)

            preds = predict(
                model=model,
                data_loader=self.test_data_loader,
                scaler=scaler,
                return_unc_parameters=False,
            )
            if i == 0:
                sum_preds = np.array(preds)
                if self.individual_ensemble_predictions:
                    individual_preds = np.expand_dims(np.array(preds), axis=-1)
            else:
                sum_preds += np.array(preds)
                if self.individual_ensemble_predictions:
                    individual_preds = np.append(
                        individual_preds, np.expand_dims(preds, axis=-1), axis=-1
                    )

        self.uncal_preds = (sum_preds / self.num_models).tolist()
        uncal_vars = np.zeros_like(sum_preds)
        uncal_vars[:] = np.nan
        self.uncal_vars = uncal_vars
        if self.individual_ensemble_predictions:
            self.individual_preds = individual_preds.tolist()

    def get_uncal_output(self):
        return self.uncal_vars


class ConformalQuantileRegressionPredictor(UncertaintyPredictor):
    """
    Class that is used for conformal quantile regression. Reformats preds to be midpoint
    of interval and outputs the interval as the uncal_output.
    """

    @property
    def label(self):
        return "no_uncertainty_method"

    def raise_argument_errors(self):
        super().raise_argument_errors()
        if self.dataset_type != "regression":
            raise ValueError(
                "Conformal quantile regression is only compatible with regression dataset types."
            )

    @staticmethod
    def reformat_preds(preds):
        """
        Reformat preds so to midpoint of quantiles
        """

        (num_data, num_tasks) = preds.shape

        preds_new = np.zeros((num_data, num_tasks//2))

        for task_id in range(num_tasks//2):
            preds_new[:, task_id] = preds[:, task_id] + preds[:, task_id + num_tasks//2]
            preds_new[:, task_id] = preds_new[:, task_id]/2

        return preds_new

    @staticmethod
    def make_intervals(preds):
        """
        Make uncalibrated intervals from the uncalibrated predictions.
        """

        return preds[:].tolist()

    def calculate_predictions(self):
        for i, (model, scaler_list) in enumerate(
            tqdm(zip(self.models, self.scalers), total=self.num_models)
        ):
            (scaler, features_scaler, atom_descriptor_scaler, bond_feature_scaler) = scaler_list
            if (
                features_scaler is not None
                or atom_descriptor_scaler is not None
                or bond_feature_scaler is not None
            ):
                self.test_data.reset_features_and_targets()
                if features_scaler is not None:
                    self.test_data.normalize_features(features_scaler)
                if atom_descriptor_scaler is not None:
                    self.test_data.normalize_features(
                        atom_descriptor_scaler, scale_atom_descriptors=True
                    )
                if bond_feature_scaler is not None:
                    self.test_data.normalize_features(bond_feature_scaler, scale_bond_features=True)

            preds = predict(
                model=model,
                data_loader=self.test_data_loader,
                scaler=scaler,
                return_unc_parameters=False,
            )
            if i == 0:
                sum_preds = np.array(preds)
                if self.individual_ensemble_predictions:
                    individual_preds = np.expand_dims(np.array(preds), axis=-1)
            else:
                sum_preds += np.array(preds)
                if self.individual_ensemble_predictions:
                    individual_preds = np.append(
                        individual_preds, np.expand_dims(preds, axis=-1), axis=-1
                    )

        self.uncal_preds = (sum_preds / self.num_models)
        self.uncal_intervals = self.make_intervals(self.uncal_preds)
        if self.individual_ensemble_predictions:
            self.individual_preds = individual_preds.tolist()

        self.uncal_preds = self.reformat_preds(self.uncal_preds).tolist()

    def get_uncal_output(self):
        return self.uncal_intervals


class ConformalRegressionPredictor(ConformalQuantileRegressionPredictor):
    """
    Class that is used for basic conformal regression. Reformats preds to be midpoint
    of interval and outputs uncalibrated interval size 0 as the uncal_output.
    """

    @staticmethod
    def reformat_preds(preds):
        """
        Reformat preds so to midpoint of quantiles
        """

        return preds

    @staticmethod
    def make_intervals(preds):
        """
        Make uncalibrated intervals from the uncalibrated predictions.
        """

        intervals = np.concatenate((preds[:], preds[:]), axis=1)

        return intervals.tolist()


class RoundRobinSpectraPredictor(UncertaintyPredictor):
    """
    A class predicting uncertainty for spectra outputs from an ensemble of models. Output is
    the average SID calculated pairwise between each of the individual spectrum predictions.
    """

    @property
    def label(self):
        return "roundrobin_sid"

    def raise_argument_errors(self):
        super().raise_argument_errors()
        if self.num_models == 1:
            raise ValueError(
                "Roundrobin uncertainty is only available when multiple models are provided."
            )

    def calculate_predictions(self):
        for i, (model, scaler_list) in enumerate(
            tqdm(zip(self.models, self.scalers), total=self.num_models)
        ):
            (scaler, features_scaler, atom_descriptor_scaler, bond_feature_scaler) = scaler_list
            if (
                features_scaler is not None
                or atom_descriptor_scaler is not None
                or bond_feature_scaler is not None
            ):
                self.test_data.reset_features_and_targets()
                if features_scaler is not None:
                    self.test_data.normalize_features(features_scaler)
                if atom_descriptor_scaler is not None:
                    self.test_data.normalize_features(
                        atom_descriptor_scaler, scale_atom_descriptors=True
                    )
                if bond_feature_scaler is not None:
                    self.test_data.normalize_features(bond_feature_scaler, scale_bond_features=True)

            preds = predict(
                model=model,
                data_loader=self.test_data_loader,
                scaler=scaler,
                return_unc_parameters=False,
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
                individual_preds = np.expand_dims(np.array(preds), axis=-1)
            else:
                sum_preds += np.array(preds)
                individual_preds = np.append(
                    individual_preds, np.expand_dims(preds, axis=-1), axis=-1
                )  # shape(data, tasks, ensemble)

        self.uncal_preds = (sum_preds / self.num_models).tolist()
        self.uncal_sid = roundrobin_sid(individual_preds)  # shape(data)
        if self.individual_ensemble_predictions:
            self.individual_preds = individual_preds.tolist()

    def get_uncal_output(self):
        return self.uncal_sid


class MVEPredictor(UncertaintyPredictor):
    """
    Class that uses the variance output of the mve loss function (aka heteroscedastic loss)
    as a prediction uncertainty.
    """

    @property
    def label(self):
        return "mve_uncal_var"

    def raise_argument_errors(self):
        super().raise_argument_errors()
        if self.loss_function != "mve":
            raise ValueError(
                "In order to use mve uncertainty, trained models must have used mve loss function."
            )

    def calculate_predictions(self):
        for i, (model, scaler_list) in enumerate(
            tqdm(zip(self.models, self.scalers), total=self.num_models)
        ):
            (scaler, features_scaler, atom_descriptor_scaler, bond_feature_scaler) = scaler_list
            if (
                features_scaler is not None
                or atom_descriptor_scaler is not None
                or bond_feature_scaler is not None
            ):
                self.test_data.reset_features_and_targets()
                if features_scaler is not None:
                    self.test_data.normalize_features(features_scaler)
                if atom_descriptor_scaler is not None:
                    self.test_data.normalize_features(
                        atom_descriptor_scaler, scale_atom_descriptors=True
                    )
                if bond_feature_scaler is not None:
                    self.test_data.normalize_features(bond_feature_scaler, scale_bond_features=True)

            preds, var = predict(
                model=model,
                data_loader=self.test_data_loader,
                scaler=scaler,
                return_unc_parameters=True,
            )
            if i == 0:
                sum_preds = np.array(preds)
                sum_squared = np.square(preds)
                sum_vars = np.array(var)
                individual_vars = [var]
                if self.individual_ensemble_predictions:
                    individual_preds = np.expand_dims(np.array(preds), axis=-1)
            else:
                sum_preds += np.array(preds)
                sum_squared += np.square(preds)
                sum_vars += np.array(var)
                individual_vars.append(var)
                if self.individual_ensemble_predictions:
                    individual_preds = np.append(
                        individual_preds, np.expand_dims(preds, axis=-1), axis=-1
                    )

        uncal_preds = sum_preds / self.num_models
        uncal_vars = (sum_vars + sum_squared) / self.num_models - np.square(
            sum_preds / self.num_models
        )
        self.uncal_preds, self.uncal_vars = uncal_preds.tolist(), uncal_vars.tolist()
        self.individual_vars = individual_vars
        if self.individual_ensemble_predictions:
            self.individual_preds = individual_preds.tolist()

    def get_uncal_output(self):
        return self.uncal_vars


class EvidentialTotalPredictor(UncertaintyPredictor):
    """
    Uses the evidential loss function to calculate total uncertainty variance from
    ancilliary loss function outputs. As presented in https://doi.org/10.1021/acscentsci.1c00546.
    """

    @property
    def label(self):
        return "evidential_total_uncal_var"

    def raise_argument_errors(self):
        super().raise_argument_errors()
        if self.loss_function != "evidential":
            raise ValueError(
                "In order to use evidential uncertainty, trained models must have used evidential regression loss function."
            )
        if self.dataset_type != "regression":
            raise ValueError(
                "Evidential total uncertainty is only compatible with regression dataset types."
            )

    def calculate_predictions(self):
        for i, (model, scaler_list) in enumerate(
            tqdm(zip(self.models, self.scalers), total=self.num_models)
        ):
            (scaler, features_scaler, atom_descriptor_scaler, bond_feature_scaler) = scaler_list
            if (
                features_scaler is not None
                or atom_descriptor_scaler is not None
                or bond_feature_scaler is not None
            ):
                self.test_data.reset_features_and_targets()
                if features_scaler is not None:
                    self.test_data.normalize_features(features_scaler)
                if atom_descriptor_scaler is not None:
                    self.test_data.normalize_features(
                        atom_descriptor_scaler, scale_atom_descriptors=True
                    )
                if bond_feature_scaler is not None:
                    self.test_data.normalize_features(bond_feature_scaler, scale_bond_features=True)

            preds, lambdas, alphas, betas = predict(
                model=model,
                data_loader=self.test_data_loader,
                scaler=scaler,
                return_unc_parameters=True,
            )
            var = np.array(betas) * (1 + 1 / np.array(lambdas)) / (np.array(alphas) - 1)
            if i == 0:
                sum_preds = np.array(preds)
                sum_squared = np.square(preds)
                sum_vars = np.array(var)
                individual_vars = [var]
                if self.individual_ensemble_predictions:
                    individual_preds = np.expand_dims(np.array(preds), axis=-1)
            else:
                sum_preds += np.array(preds)
                sum_squared += np.square(preds)
                sum_vars += np.array(var)
                individual_vars.append(var)
                if self.individual_ensemble_predictions:
                    individual_preds = np.append(
                        individual_preds, np.expand_dims(preds, axis=-1), axis=-1
                    )

        uncal_preds = sum_preds / self.num_models
        uncal_vars = (sum_vars + sum_squared) / self.num_models - np.square(
            sum_preds / self.num_models
        )
        self.uncal_preds, self.uncal_vars = uncal_preds.tolist(), uncal_vars.tolist()
        self.individual_vars = individual_vars
        if self.individual_ensemble_predictions:
            self.individual_preds = individual_preds.tolist()

    def get_uncal_output(self):
        return self.uncal_vars


class EvidentialAleatoricPredictor(UncertaintyPredictor):
    """
    Uses the evidential loss function to calculate aleatoric uncertainty variance from
    ancilliary loss function outputs. As presented in https://doi.org/10.1021/acscentsci.1c00546.
    """

    @property
    def label(self):
        return "evidential_aleatoric_uncal_var"

    def raise_argument_errors(self):
        super().raise_argument_errors()
        if self.loss_function != "evidential":
            raise ValueError(
                "In order to use evidential uncertainty, trained models must have used evidential regression loss function."
            )
        if self.dataset_type != "regression":
            raise ValueError(
                "Evidential aleatoric uncertainty is only compatible with regression dataset types."
            )

    def calculate_predictions(self):
        for i, (model, scaler_list) in enumerate(
            tqdm(zip(self.models, self.scalers), total=self.num_models)
        ):
            (scaler, features_scaler, atom_descriptor_scaler, bond_feature_scaler) = scaler_list
            if (
                features_scaler is not None
                or atom_descriptor_scaler is not None
                or bond_feature_scaler is not None
            ):
                self.test_data.reset_features_and_targets()
                if features_scaler is not None:
                    self.test_data.normalize_features(features_scaler)
                if atom_descriptor_scaler is not None:
                    self.test_data.normalize_features(
                        atom_descriptor_scaler, scale_atom_descriptors=True
                    )
                if bond_feature_scaler is not None:
                    self.test_data.normalize_features(bond_feature_scaler, scale_bond_features=True)

            preds, lambdas, alphas, betas = predict(
                model=model,
                data_loader=self.test_data_loader,
                scaler=scaler,
                return_unc_parameters=True,
            )
            var = np.array(betas) / (np.array(alphas) - 1)
            if i == 0:
                sum_preds = np.array(preds)
                sum_squared = np.square(preds)
                sum_vars = np.array(var)
                individual_vars = [var]
                if self.individual_ensemble_predictions:
                    individual_preds = np.expand_dims(np.array(preds), axis=-1)
            else:
                sum_preds += np.array(preds)
                sum_squared += np.square(preds)
                sum_vars += np.array(var)
                individual_vars.append(var)
                if self.individual_ensemble_predictions:
                    individual_preds = np.append(
                        individual_preds, np.expand_dims(preds, axis=-1), axis=-1
                    )

        uncal_preds = sum_preds / self.num_models
        uncal_vars = (sum_vars + sum_squared) / self.num_models - np.square(
            sum_preds / self.num_models
        )
        self.uncal_preds, self.uncal_vars = uncal_preds.tolist(), uncal_vars.tolist()
        self.individual_vars = individual_vars
        if self.individual_ensemble_predictions:
            self.individual_preds = individual_preds.tolist()

    def get_uncal_output(self):
        return self.uncal_vars


class EvidentialEpistemicPredictor(UncertaintyPredictor):
    """
    Uses the evidential loss function to calculate epistemic uncertainty variance from
    ancilliary loss function outputs. As presented in https://doi.org/10.1021/acscentsci.1c00546.
    """

    @property
    def label(self):
        return "evidential_epistemic_uncal_var"

    def raise_argument_errors(self):
        super().raise_argument_errors()
        if self.loss_function != "evidential":
            raise ValueError(
                "In order to use evidential uncertainty, trained models must have used evidential regression loss function."
            )
        if self.dataset_type != "regression":
            raise ValueError(
                "Evidential epistemic uncertainty is only compatible with regression dataset types."
            )

    def calculate_predictions(self):
        for i, (model, scaler_list) in enumerate(
            tqdm(zip(self.models, self.scalers), total=self.num_models)
        ):
            (scaler, features_scaler, atom_descriptor_scaler, bond_feature_scaler) = scaler_list
            if (
                features_scaler is not None
                or atom_descriptor_scaler is not None
                or bond_feature_scaler is not None
            ):
                self.test_data.reset_features_and_targets()
                if features_scaler is not None:
                    self.test_data.normalize_features(features_scaler)
                if atom_descriptor_scaler is not None:
                    self.test_data.normalize_features(
                        atom_descriptor_scaler, scale_atom_descriptors=True
                    )
                if bond_feature_scaler is not None:
                    self.test_data.normalize_features(bond_feature_scaler, scale_bond_features=True)

            preds, lambdas, alphas, betas = predict(
                model=model,
                data_loader=self.test_data_loader,
                scaler=scaler,
                return_unc_parameters=True,
            )
            var = np.array(betas) / (np.array(lambdas) * (np.array(alphas) - 1))
            if i == 0:
                sum_preds = np.array(preds)
                sum_squared = np.square(preds)
                sum_vars = np.array(var)
                individual_vars = [var]
                if self.individual_ensemble_predictions:
                    individual_preds = np.expand_dims(np.array(preds), axis=-1)
            else:
                sum_preds += np.array(preds)
                sum_squared += np.square(preds)
                sum_vars += np.array(var)
                individual_vars.append(var)
                if self.individual_ensemble_predictions:
                    individual_preds = np.append(
                        individual_preds, np.expand_dims(preds, axis=-1), axis=-1
                    )

        uncal_preds = sum_preds / self.num_models
        uncal_vars = (sum_vars + sum_squared) / self.num_models - np.square(
            sum_preds / self.num_models
        )
        self.uncal_preds, self.uncal_vars = uncal_preds.tolist(), uncal_vars.tolist()
        self.individual_vars = individual_vars
        if self.individual_ensemble_predictions:
            self.individual_preds = individual_preds.tolist()

    def get_uncal_output(self):
        return self.uncal_vars


class EnsemblePredictor(UncertaintyPredictor):
    """
    Class that predicts uncertainty for predictions based on the variance in predictions among
    an ensemble's submodels.
    """

    @property
    def label(self):
        return "ensemble_uncal_var"

    def raise_argument_errors(self):
        super().raise_argument_errors()
        if self.num_models == 1:
            raise ValueError(
                "Ensemble method for uncertainty is only available when multiple models are provided."
            )

    def calculate_predictions(self):
        for i, (model, scaler_list) in enumerate(
            tqdm(zip(self.models, self.scalers), total=self.num_models)
        ):
            (scaler, features_scaler, atom_descriptor_scaler, bond_feature_scaler) = scaler_list
            if (
                features_scaler is not None
                or atom_descriptor_scaler is not None
                or bond_feature_scaler is not None
            ):
                self.test_data.reset_features_and_targets()
                if features_scaler is not None:
                    self.test_data.normalize_features(features_scaler)
                if atom_descriptor_scaler is not None:
                    self.test_data.normalize_features(
                        atom_descriptor_scaler, scale_atom_descriptors=True
                    )
                if bond_feature_scaler is not None:
                    self.test_data.normalize_features(bond_feature_scaler, scale_bond_features=True)
            preds = predict(
                model=model,
                data_loader=self.test_data_loader,
                scaler=scaler,
                return_unc_parameters=False,
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
                sum_squared = np.square(preds)
                if self.individual_ensemble_predictions:
                    individual_preds = np.expand_dims(np.array(preds), axis=-1)
                if model.train_class_sizes is not None:
                    self.train_class_sizes = [model.train_class_sizes]
            else:
                sum_preds += np.array(preds)
                sum_squared += np.square(preds)
                if self.individual_ensemble_predictions:
                    individual_preds = np.append(
                        individual_preds, np.expand_dims(preds, axis=-1), axis=-1
                    )
                if model.train_class_sizes is not None:
                    self.train_class_sizes.append(model.train_class_sizes)

        uncal_preds = sum_preds / self.num_models
        uncal_vars = sum_squared / self.num_models - np.square(sum_preds) / self.num_models**2
        self.uncal_preds, self.uncal_vars = uncal_preds.tolist(), uncal_vars.tolist()
        if self.individual_ensemble_predictions:
            self.individual_preds = individual_preds.tolist()

    def get_uncal_output(self):
        return self.uncal_vars


class DropoutPredictor(UncertaintyPredictor):
    """
    Class that creates an artificial ensemble of models by applying monte carlo dropout to the loaded
    model parameters. Predicts uncertainty for predictions based on the variance in predictions among
    an ensemble's submodels.
    """

    @property
    def label(self):
        return "dropout_uncal_var"

    def raise_argument_errors(self):
        super().raise_argument_errors()
        if self.num_models > 1:
            raise ValueError(
                "Dropout method for uncertainty should be used for a single model rather than an ensemble."
            )

    def calculate_predictions(self):
        model = next(self.models)
        (scaler, features_scaler, atom_descriptor_scaler, bond_feature_scaler) = next(self.scalers)
        if (
            features_scaler is not None
            or atom_descriptor_scaler is not None
            or bond_feature_scaler is not None
        ):
            self.test_data.reset_features_and_targets()
            if features_scaler is not None:
                self.test_data.normalize_features(features_scaler)
            if atom_descriptor_scaler is not None:
                self.test_data.normalize_features(
                    atom_descriptor_scaler, scale_atom_descriptors=True
                )
            if bond_feature_scaler is not None:
                self.test_data.normalize_features(bond_feature_scaler, scale_bond_features=True)
        for i in range(self.dropout_sampling_size):
            preds = predict(
                model=model,
                data_loader=self.test_data_loader,
                scaler=scaler,
                return_unc_parameters=False,
                dropout_prob=self.uncertainty_dropout_p,
            )
            if i == 0:
                sum_preds = np.array(preds)
                sum_squared = np.square(preds)
            else:
                sum_preds += np.array(preds)
                sum_squared += np.square(preds)

        uncal_preds = sum_preds / self.dropout_sampling_size
        uncal_vars = (
            sum_squared / self.dropout_sampling_size
            - np.square(sum_preds) / self.dropout_sampling_size**2
        )
        self.uncal_preds, self.uncal_vars = uncal_preds.tolist(), uncal_vars.tolist()

    def get_uncal_output(self):
        return self.uncal_vars


class ClassPredictor(UncertaintyPredictor):
    """
    Class uses the [0,1] range of results from classification or multiclass models
    as the indicator of confidence. Used for classification and multiclass dataset types.
    """

    @property
    def label(self):
        return "classification_uncal_confidence"

    def raise_argument_errors(self):
        super().raise_argument_errors()
        if self.dataset_type not in ["classification", "multiclass"]:
            raise ValueError(
                "Classification output uncertainty method must be used with dataset types classification or multiclass."
            )

    def calculate_predictions(self):
        for i, (model, scaler_list) in enumerate(
            tqdm(zip(self.models, self.scalers), total=self.num_models)
        ):
            (scaler, features_scaler, atom_descriptor_scaler, bond_feature_scaler) = scaler_list
            if (
                features_scaler is not None
                or atom_descriptor_scaler is not None
                or bond_feature_scaler is not None
            ):
                self.test_data.reset_features_and_targets()
                if features_scaler is not None:
                    self.test_data.normalize_features(features_scaler)
                if atom_descriptor_scaler is not None:
                    self.test_data.normalize_features(
                        atom_descriptor_scaler, scale_atom_descriptors=True
                    )
                if bond_feature_scaler is not None:
                    self.test_data.normalize_features(bond_feature_scaler, scale_bond_features=True)

            preds = predict(
                model=model,
                data_loader=self.test_data_loader,
                scaler=scaler,
                return_unc_parameters=False,
            )
            if i == 0:
                sum_preds = np.array(preds)
                if self.individual_ensemble_predictions:
                    individual_preds = np.expand_dims(np.array(preds), axis=-1)
                if model.train_class_sizes is not None:
                    self.train_class_sizes = [model.train_class_sizes]
            else:
                sum_preds += np.array(preds)
                if self.individual_ensemble_predictions:
                    individual_preds = np.append(
                        individual_preds, np.expand_dims(preds, axis=-1), axis=-1
                    )
                if model.train_class_sizes is not None:
                    self.train_class_sizes.append(model.train_class_sizes)

        self.uncal_preds = (sum_preds / self.num_models).tolist()
        self.uncal_confidence = self.uncal_preds
        if self.individual_ensemble_predictions:
            self.individual_preds = individual_preds.tolist()

    def get_uncal_output(self):
        return self.uncal_confidence


def build_uncertainty_predictor(
    uncertainty_method: str,
    test_data: MoleculeDataset,
    test_data_loader: MoleculeDataLoader,
    models: Iterator[MoleculeModel],
    scalers: Iterator[StandardScaler],
    num_models: int,
    dataset_type: str,
    loss_function: str,
    uncertainty_dropout_p: float,
    conformal_alpha: float,
    dropout_sampling_size: int,
    individual_ensemble_predictions: bool,
    spectra_phase_mask: List[List[bool]],
) -> UncertaintyPredictor:
    """
    Function that chooses and returns the appropriate :class: `UncertaintyPredictor` subclass
    for the provided arguments.
    """

    supported_predictors = {
        None: NoUncertaintyPredictor,
        "mve": MVEPredictor,
        "ensemble": EnsemblePredictor,
        "classification": ClassPredictor,
        "evidential_total": EvidentialTotalPredictor,
        "evidential_epistemic": EvidentialEpistemicPredictor,
        "evidential_aleatoric": EvidentialAleatoricPredictor,
        "dropout": DropoutPredictor,
        "spectra_roundrobin": RoundRobinSpectraPredictor,
        "conformal_quantile_regression": ConformalQuantileRegressionPredictor,
        "conformal_regression": ConformalRegressionPredictor,
    }

    predictor_class = supported_predictors.get(uncertainty_method, None)

    if predictor_class is None:
        raise NotImplementedError(
            f"Uncertainty predictor type {uncertainty_method} is not currently supported. Avalable options are: {list(supported_predictors.keys())}"
        )
    else:
        predictor = predictor_class(
            test_data=test_data,
            test_data_loader=test_data_loader,
            models=models,
            scalers=scalers,
            num_models=num_models,
            dataset_type=dataset_type,
            loss_function=loss_function,
            uncertainty_dropout_p=uncertainty_dropout_p,
            conformal_alpha=conformal_alpha,
            dropout_sampling_size=dropout_sampling_size,
            individual_ensemble_predictions=individual_ensemble_predictions,
            spectra_phase_mask=spectra_phase_mask,
        )
    return predictor
