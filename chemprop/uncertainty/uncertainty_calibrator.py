from abc import ABC, abstractmethod
from typing import Iterator, List

import numpy as np
from chemprop.data.data import MoleculeDataLoader
from scipy.special import erfinv, softmax, logit, expit
from scipy.optimize import fmin
from scipy.stats import t
from sklearn.isotonic import IsotonicRegression

from chemprop.data import MoleculeDataset, StandardScaler
from chemprop.models import MoleculeModel
from chemprop.uncertainty.uncertainty_predictor import build_uncertainty_predictor, UncertaintyPredictor
from chemprop.multitask_utils import reshape_values


class UncertaintyCalibrator(ABC):
    """
    Uncertainty calibrator class. Subclasses for each uncertainty calibration
    method. Subclasses should override the calibrate and apply functions for
    implemented metrics.
    """

    def __init__(
        self,
        uncertainty_method: str,
        interval_percentile: int,
        regression_calibrator_metric: str,
        calibration_data: MoleculeDataset,
        calibration_data_loader: MoleculeDataLoader,
        models: Iterator[MoleculeModel],
        scalers: Iterator[StandardScaler],
        num_models: int,
        dataset_type: str,
        loss_function: str,
        uncertainty_dropout_p: float,
        dropout_sampling_size: int,
        spectra_phase_mask: List[List[bool]],
    ):
        self.calibration_data = calibration_data
        self.calibration_data_loader = calibration_data_loader
        self.regression_calibrator_metric = regression_calibrator_metric
        self.interval_percentile = interval_percentile
        self.dataset_type = dataset_type
        self.uncertainty_method = uncertainty_method
        self.loss_function = loss_function
        self.num_models = num_models

        self.raise_argument_errors()

        self.calibration_predictor = build_uncertainty_predictor(
            test_data=calibration_data,
            test_data_loader=calibration_data_loader,
            models=models,
            scalers=scalers,
            num_models=num_models,
            dataset_type=dataset_type,
            loss_function=loss_function,
            uncertainty_method=uncertainty_method,
            uncertainty_dropout_p=uncertainty_dropout_p,
            dropout_sampling_size=dropout_sampling_size,
            individual_ensemble_predictions=False,
            spectra_phase_mask=spectra_phase_mask,
        )

        self.calibrate()

    @property
    @abstractmethod
    def label(self):
        """
        The string in saved results indicating the uncertainty method used.
        """

    def raise_argument_errors(self):
        """
        Raise errors for incompatibilities between dataset type and uncertainty method, or similar.
        """
        if self.dataset_type == "spectra":
            raise NotImplementedError(
                "No uncertainty calibrators are implemented for the spectra dataset type."
            )
        if self.uncertainty_method in ["ensemble", "dropout"] and self.dataset_type in ["classification", "multiclass"]:
            raise NotImplementedError(
                "Though ensemble and dropout uncertainty methods are available for classification \
                    multiclass dataset types, their outputs are not confidences and are not \
                    compatible with any implemented calibration methods for classification."
            )

    @abstractmethod
    def calibrate(self):
        """
        Fit calibration method for the calibration data.
        """

    @abstractmethod
    def apply_calibration(self, uncal_predictor: UncertaintyPredictor):
        """
        Take in predictions and uncertainty parameters from a model and apply the calibration method using fitted parameters.
        """

    @abstractmethod
    def nll(
        self,
        preds: List[List[float]],
        unc: List[List[float]],
        targets: List[List[float]],
        mask: List[List[bool]],
    ) -> List[float]:
        """
        Takes in calibrated predictions and uncertainty parameters and returns the log probability density of that result.
        """


class ZScalingCalibrator(UncertaintyCalibrator):
    """
    A class that calibrates regression uncertainty models by applying
    a scaling value to the uncalibrated standard deviation, fitted by minimizing the
    negative log likelihood of a normal distribution around each prediction
    with scaling given by the uncalibrated variance. Method is described
    in https://arxiv.org/abs/1905.11659.
    """

    @property
    def label(self):
        if self.regression_calibrator_metric == "stdev":
            label = f"{self.uncertainty_method}_zscaling_stdev"
        else:  # interval
            label = (
                f"{self.uncertainty_method}_zscaling_{self.interval_percentile}interval"
            )
        return label

    def raise_argument_errors(self):
        super().raise_argument_errors()
        if self.dataset_type != "regression":
            raise ValueError(
                "Z Score Scaling is only compatible with regression datasets."
            )

    def calibrate(self):
        uncal_preds = np.array(
            self.calibration_predictor.get_uncal_preds()
        )  # shape(data, tasks)
        uncal_vars = np.array(self.calibration_predictor.get_uncal_vars())
        targets = np.array(self.calibration_data.targets())
        mask = np.array(self.calibration_data.mask())
        self.num_tasks = len(mask)
        if self.calibration_data.is_atom_bond_targets:
            uncal_preds = [np.concatenate(x) for x in zip(*uncal_preds)]
            uncal_vars = [np.concatenate(x) for x in zip(*uncal_vars)]
            targets = [np.concatenate(x) for x in zip(*targets)]
        else:
            uncal_preds = np.array(list(zip(*uncal_preds)))
            uncal_vars = np.array(list(zip(*uncal_vars)))
            targets = targets.astype(float)
            targets = np.array(list(zip(*targets)))
        self.scaling = np.zeros(self.num_tasks)

        for i in range(self.num_tasks):
            task_mask = mask[i]
            task_targets = targets[i][task_mask]
            task_preds = uncal_preds[i][task_mask]
            task_vars = uncal_vars[i][task_mask]
            task_errors = task_preds - task_targets
            task_zscore = task_errors / np.sqrt(task_vars)

            def objective(scaler_value: float):
                scaled_vars = task_vars * scaler_value ** 2
                nll = np.log(2 * np.pi * scaled_vars) / 2 \
                    + (task_errors) ** 2 / (2 * scaled_vars)
                return nll.sum()

            initial_guess = np.std(task_zscore)
            sol = fmin(objective, initial_guess)

            if self.regression_calibrator_metric == "stdev":
                self.scaling[i] = sol
            else:  # interval
                self.scaling[i] = (
                    sol * erfinv(self.interval_percentile / 100) * np.sqrt(2)
                )

    def apply_calibration(self, uncal_predictor: UncertaintyPredictor):
        uncal_preds = np.array(uncal_predictor.get_uncal_preds())
        uncal_vars = np.array(uncal_predictor.get_uncal_vars())
        if self.calibration_data.is_atom_bond_targets:
            cal_stdev = []
            sqrt_uncal_vars = [
                [np.sqrt(var) for var in uncal_var] for uncal_var in uncal_vars
            ]
            for sqrt_uncal_var in sqrt_uncal_vars:
                scaled_stdev = [var * s for var, s in zip(sqrt_uncal_var, self.scaling)]
                cal_stdev.append(scaled_stdev)
            return uncal_preds, cal_stdev
        else:
            cal_stdev = np.sqrt(uncal_vars) * np.expand_dims(self.scaling, axis=0)
            return uncal_preds.tolist(), cal_stdev.tolist()

    def nll(
        self,
        preds: List[List[float]],
        unc: List[List[float]],
        targets: List[List[float]],
        mask: List[List[bool]],
    ):
        unc_var = np.square(unc)
        preds = np.array(preds)
        targets = np.array(targets)
        mask = np.array(mask)
        if self.calibration_data.is_atom_bond_targets:
            unc_var = [np.concatenate(x) for x in zip(*unc_var)]
            preds = [np.concatenate(x) for x in zip(*preds)]
            targets = [np.concatenate(x) for x in zip(*targets)]
        else:
            unc_var = np.array(list(zip(*unc_var)))
            preds = np.array(list(zip(*preds)))
            targets = targets.astype(float)
            targets = np.array(list(zip(*targets)))
        nll = []
        for i in range(self.num_tasks):
            task_mask = mask[i]
            task_preds = preds[i][task_mask]
            task_targets = targets[i][task_mask]
            task_unc = unc_var[i][task_mask]
            task_nll = (
                np.log(2 * np.pi * task_unc) / 2
                + (task_preds - task_targets) ** 2 / (2 * task_unc)
            )
            nll.append(task_nll.mean())
        return nll


class TScalingCalibrator(UncertaintyCalibrator):
    """
    A class that calibrates regression uncertainty models using a variation of the
    ZScaling method. Instead, this method assumes that error is dominated by
    variance error as represented by the variance of the ensemble predictions.
    The scaling value is obtained by minimizing the negative log likelihood
    of the t distribution, including reductio term due to the number of ensemble models sampled.
    """

    @property
    def label(self):
        if self.regression_calibrator_metric == "stdev":
            label = f"{self.uncertainty_method}_tscaling_stdev"
        else:  # interval
            label = (
                f"{self.uncertainty_method}_tscaling_{self.interval_percentile}interval"
            )
        return label

    def raise_argument_errors(self):
        super().raise_argument_errors()
        if self.dataset_type != "regression":
            raise ValueError(
                "T Score Scaling is only compatible with regression datasets."
            )
        if self.uncertainty_method == "dropout":
            raise ValueError(
                "T scaling not enabled with dropout variance uncertainty method."
            )
        if self.num_models == 1:
            raise ValueError("T scaling is intended for use with ensemble models.")

    def calibrate(self):
        uncal_preds = np.array(
            self.calibration_predictor.get_uncal_preds()
        )  # shape(data, tasks)
        uncal_vars = np.array(self.calibration_predictor.get_uncal_vars())
        targets = np.array(self.calibration_data.targets())
        mask = np.array(self.calibration_data.mask())
        self.num_tasks = len(mask)
        if self.calibration_data.is_atom_bond_targets:
            uncal_preds = [np.concatenate(x) for x in zip(*uncal_preds)]
            uncal_vars = [np.concatenate(x) for x in zip(*uncal_vars)]
            targets = [np.concatenate(x) for x in zip(*targets)]
        else:
            uncal_preds = np.array(list(zip(*uncal_preds)))
            uncal_vars = np.array(list(zip(*uncal_vars)))
            targets = targets.astype(float)
            targets = np.array(list(zip(*targets)))
        self.scaling = np.zeros(self.num_tasks)

        for i in range(self.num_tasks):
            task_mask = mask[i]
            task_targets = targets[i][task_mask]
            task_preds = uncal_preds[i][task_mask]
            task_vars = uncal_vars[i][task_mask]
            std_error_of_mean = np.sqrt(
                task_vars / (self.num_models - 1)
            )  # reduced for number of samples and include Bessel's correction
            task_errors = task_preds - task_targets
            task_tscore = task_errors / std_error_of_mean

            def objective(scaler_value: np.ndarray):
                scaled_std = std_error_of_mean * scaler_value
                likelihood = t.pdf(
                    x=task_errors, df=self.num_models - 1, scale=scaled_std
                )  # scipy t distribution pdf
                nll = -1 * np.sum(np.log(likelihood), axis=0)
                return nll

            initial_guess = np.std(task_tscore)
            stdev_scaling = fmin(objective, initial_guess)
            if self.regression_calibrator_metric == "stdev":
                self.scaling[i] = stdev_scaling
            else:  # interval
                interval_scaling = stdev_scaling * t.ppf(
                    (self.interval_percentile / 100 + 1) / 2, df=self.num_models - 1
                )
                self.scaling[i] = interval_scaling

    def apply_calibration(self, uncal_predictor: UncertaintyPredictor):
        uncal_preds = np.array(uncal_predictor.get_uncal_preds())
        uncal_vars = np.array(uncal_predictor.get_uncal_vars())
        if self.calibration_data.is_atom_bond_targets:
            cal_stdev = []
            sqrt_uncal_vars = [
                [np.sqrt(var / (self.num_models - 1)) for var in uncal_var]
                for uncal_var in uncal_vars
            ]
            for sqrt_uncal_var in sqrt_uncal_vars:
                scaled_stdev = [var * s for var, s in zip(sqrt_uncal_var, self.scaling)]
                cal_stdev.append(scaled_stdev)
            return uncal_preds, cal_stdev
        else:
            cal_stdev = np.sqrt(uncal_vars / (self.num_models - 1)) * np.expand_dims(
                self.scaling, axis=0
            )
        return uncal_preds.tolist(), cal_stdev.tolist()

    def nll(
        self,
        preds: List[List[float]],
        unc: List[List[float]],
        targets: List[List[float]],
        mask: List[List[bool]],
    ):
        unc = np.square(unc)
        preds = np.array(preds)
        targets = np.array(targets)
        mask = np.array(mask)
        if self.calibration_data.is_atom_bond_targets:
            unc = [np.concatenate(x) for x in zip(*unc)]
            preds = [np.concatenate(x) for x in zip(*preds)]
            targets = [np.concatenate(x) for x in zip(*targets)]
        else:
            unc = np.array(list(zip(*unc)))
            preds = np.array(list(zip(*preds)))
            targets = targets.astype(float)
            targets = np.array(list(zip(*targets)))
        nll = []
        for i in range(self.num_tasks):
            task_mask = mask[i]
            task_preds = preds[i][task_mask]
            task_targets = targets[i][task_mask]
            task_unc = unc[i][task_mask]
            task_nll = -1 * t.logpdf(
                x=task_preds - task_targets, scale=task_unc, df=self.num_models - 1
            )
            nll.append(task_nll.mean())
        return nll


class ZelikmanCalibrator(UncertaintyCalibrator):
    """
    A calibrator for regression datasets that does not depend on a particular probability
    function form. Designed to be used with interval output. Uses the "CRUDE" method as
    described in https://arxiv.org/abs/2005.12496. As implemented here, the interval
    bounds are constrained to be symmetrical, though this is not required in the source method.
    The probability density to be used for NLL evaluator for the zelikman interval method is
    approximated here as a histogram function.
    """

    @property
    def label(self):
        if self.regression_calibrator_metric == "stdev":
            label = f"{self.uncertainty_method}_zelikman_stdev"
        else:
            label = f"{self.uncertainty_method}_zelikman_{self.interval_percentile}interval"
        return label

    def raise_argument_errors(self):
        super().raise_argument_errors()
        if self.dataset_type != "regression":
            raise ValueError(
                "Crude Scaling is only compatible with regression datasets."
            )

    def calibrate(self):
        uncal_preds = np.array(
            self.calibration_predictor.get_uncal_preds()
        )  # shape(data, tasks)
        uncal_vars = np.array(self.calibration_predictor.get_uncal_vars())
        targets = np.array(self.calibration_data.targets())
        mask = np.array(self.calibration_data.mask())
        self.num_tasks = len(mask)
        if self.calibration_data.is_atom_bond_targets:
            uncal_preds = [np.concatenate(x) for x in zip(*uncal_preds)]
            uncal_vars = [np.concatenate(x) for x in zip(*uncal_vars)]
            targets = [np.concatenate(x) for x in zip(*targets)]
        else:
            uncal_preds = np.array(list(zip(*uncal_preds)))
            uncal_vars = np.array(list(zip(*uncal_vars)))
            targets = targets.astype(float)
            targets = np.array(list(zip(*targets)))
        self.histogram_parameters = []
        self.scaling = np.zeros(self.num_tasks)
        for i in range(self.num_tasks):
            task_mask = mask[i]
            task_preds = uncal_preds[i][task_mask]
            task_targets = targets[i][task_mask]
            task_vars = uncal_vars[i][task_mask]
            task_preds = np.abs(task_preds - task_targets) / np.sqrt(task_vars)
            if self.regression_calibrator_metric == "interval":
                interval_scaling = np.percentile(task_preds, self.interval_percentile)
                self.scaling[i] = interval_scaling
            else:
                symmetric_z = np.concatenate([task_preds, -1 * task_preds])
                std_scaling = np.std(symmetric_z, axis=0)
                self.scaling[i] = std_scaling
            # histogram parameters for nll calculation
            h_params = np.histogram(task_preds, bins='auto', density=True)
            self.histogram_parameters.append(h_params)

    def apply_calibration(self, uncal_predictor: UncertaintyPredictor):
        uncal_preds = np.array(uncal_predictor.get_uncal_preds())
        uncal_vars = np.array(uncal_predictor.get_uncal_vars())
        if self.calibration_data.is_atom_bond_targets:
            cal_stdev = []
            sqrt_uncal_vars = [
                [np.sqrt(var) for var in uncal_var] for uncal_var in uncal_vars
            ]
            for sqrt_uncal_var in sqrt_uncal_vars:
                scaled_stdev = [var * s for var, s in zip(sqrt_uncal_var, self.scaling)]
                cal_stdev.append(scaled_stdev)
            return uncal_preds, cal_stdev
        else:
            cal_stdev = np.sqrt(uncal_vars) * np.expand_dims(self.scaling, axis=0)
            return uncal_preds.tolist(), cal_stdev.tolist()

    def nll(
        self,
        preds: List[List[float]],
        unc: List[List[float]],
        targets: List[List[float]],
        mask: List[List[bool]],
    ):
        preds = np.array(preds)
        unc = np.array(unc)
        targets = np.array(targets)
        mask = np.array(mask)
        if self.calibration_data.is_atom_bond_targets:
            preds = [np.concatenate(x) for x in zip(*preds)]
            unc = [np.concatenate(x) for x in zip(*unc)]
            targets = [np.concatenate(x) for x in zip(*targets)]
        else:
            preds = np.array(list(zip(*preds)))
            unc = np.array(list(zip(*unc)))
            targets = targets.astype(float)
            targets = np.array(list(zip(*targets)))
        nll = []
        for i in range(self.num_tasks):
            task_mask = mask[i]
            task_preds = preds[i][task_mask]
            task_targets = targets[i][task_mask]
            task_stdev = unc[i][task_mask] / self.scaling[i]
            task_abs_z = np.abs(task_preds - task_targets) / task_stdev
            bin_edges = self.histogram_parameters[i][1]
            bin_magnitudes = self.histogram_parameters[i][0]
            bin_magnitudes = np.insert(bin_magnitudes, [0, len(bin_magnitudes)], 0)
            pred_bins = np.searchsorted(bin_edges, task_abs_z)
            # magnitude adjusted by stdev scale of the distribution and symmetry assumption
            task_likelihood = bin_magnitudes[pred_bins] / task_stdev / 2
            task_nll = -1 * np.log(task_likelihood)
            nll.append(task_nll.mean())
        return nll


class MVEWeightingCalibrator(UncertaintyCalibrator):
    """
    A method of calibration for models that have ensembles of individual models that
    make variance predictions. Minimizes the negative log likelihood for the
    predictions versus the targets by applying a weighted average across the
    variance predictions of the ensemble. Discussed in https://doi.org/10.1186/s13321-021-00551-x.
    """

    @property
    def label(self):
        if self.regression_calibrator_metric == "stdev":
            label = f"{self.uncertainty_method}_mve_weighting_stdev"
        else:  # interval
            label = f"{self.uncertainty_method}_mve_weighting_{self.interval_percentile}interval"
        return label

    def raise_argument_errors(self):
        super().raise_argument_errors()
        if self.dataset_type != "regression":
            raise ValueError(
                f"MVE Weighting is only compatible with regression datasets! got: {self.dataset_type}"
            )
        if self.loss_function not in ["mve", "evidential"]:
            raise ValueError(
                "MVE Weighting calibration can only be carried out with MVE or Evidential loss function models."
            )
        if self.num_models == 1:
            raise ValueError(
                "MVE Weighting is only useful when weighting between results in an ensemble. Only one model was provided."
            )

    def calibrate(self):
        uncal_preds = np.array(
            self.calibration_predictor.get_uncal_preds()
        )  # shape(data, tasks)
        individual_vars = np.array(
            self.calibration_predictor.get_individual_vars()
        )  # shape(models, data, tasks)
        targets = np.array(self.calibration_data.targets())
        mask = np.array(self.calibration_data.mask())
        self.num_tasks = len(mask)
        if self.calibration_data.is_atom_bond_targets:
            uncal_preds = [np.concatenate(x) for x in zip(*uncal_preds)]
            individual_vars = [np.array([np.concatenate(individual_vars[j][i][:, :]) for j in range(self.num_models)]) for i in range(self.num_tasks)]
            targets = [np.concatenate(x) for x in zip(*targets)]
        else:
            uncal_preds = np.array(list(zip(*uncal_preds)))
            individual_vars = [individual_vars[:, :, i] for i in range(self.num_tasks)]
            targets = targets.astype(float)
            targets = np.array(list(zip(*targets)))
        self.var_weighting = np.zeros([self.num_models, self.num_tasks])  # shape(models, tasks)

        for i in range(self.num_tasks):
            task_mask = mask[i]
            task_targets = targets[i][task_mask]
            task_preds = uncal_preds[i][task_mask]
            task_ind_vars = individual_vars[i][:, task_mask]
            task_errors = task_preds - task_targets

            def objective(scaler_values: np.ndarray):
                scaler_values = np.reshape(softmax(scaler_values), [-1, 1])  # (models, 1)
                scaled_vars = np.sum(
                    task_ind_vars * scaler_values, axis=0, keepdims=False
                )  # shape(data)
                nll = np.log(2 * np.pi * scaled_vars) / 2 + (task_errors) ** 2 / (
                    2 * scaled_vars
                )
                nll = np.sum(nll)
                return nll

            initial_guess = np.ones(self.num_models)
            sol = fmin(objective, initial_guess)
            self.var_weighting[:, i] = softmax(sol)
        if self.regression_calibrator_metric == "stdev":
            self.scaling = np.repeat(1, self.num_tasks)
        else:  # interval
            self.scaling = np.repeat(erfinv(self.interval_percentile / 100) * np.sqrt(2), self.num_tasks)

    def apply_calibration(self, uncal_predictor: UncertaintyPredictor):
        uncal_preds = np.array(uncal_predictor.get_uncal_preds())
        uncal_individual_vars = np.array(uncal_predictor.get_individual_vars())
        weighted_vars = None
        for ind_vars, s in zip(uncal_individual_vars, self.var_weighting):
            if weighted_vars is None:
                weighted_vars = ind_vars
                for i in range(len(s)):
                    weighted_vars[i] *= s[i]
            else:
                for i in range(len(s)):
                    weighted_vars[i] += ind_vars[i] * s[i]
        if self.calibration_data.is_atom_bond_targets:
            sqrt_weighted_vars = [np.array([np.sqrt(var) for var in uncal_var]) for uncal_var in weighted_vars]
            weighted_stdev = sqrt_weighted_vars * self.scaling
            natom_targets = len(self.calibration_data[0].atom_targets) if self.calibration_data[0].atom_targets is not None else 0
            nbond_targets = len(self.calibration_data[0].bond_targets) if self.calibration_data[0].bond_targets is not None else 0
            weighted_stdev = reshape_values(
                weighted_stdev,
                self.calibration_data,
                natom_targets,
                nbond_targets,
                len(weighted_stdev),
            )
            return uncal_preds, weighted_stdev
        else:
            weighted_stdev = np.sqrt(weighted_vars) * self.scaling
            return uncal_preds.tolist(), weighted_stdev.tolist()

    def nll(
        self,
        preds: List[List[float]],
        unc: List[List[float]],
        targets: List[List[float]],
        mask: List[List[bool]],
    ):
        unc_var = np.square(unc)
        preds = np.array(preds)
        targets = np.array(targets)
        mask = np.array(mask)
        if self.calibration_data.is_atom_bond_targets:
            unc_var = [np.concatenate(np.square(x)) for x in zip(*unc)]
            preds = [np.concatenate(x) for x in zip(*preds)]
            targets = [np.concatenate(x) for x in zip(*targets)]
        else:
            unc_var = np.array(list(zip(*unc_var)))
            preds = np.array(list(zip(*preds)))
            targets = targets.astype(float)
            targets = np.array(list(zip(*targets)))
        nll = []
        for i in range(self.num_tasks):
            task_mask = mask[i]
            task_preds = preds[i][task_mask]
            task_targets = targets[i][task_mask]
            task_unc = unc_var[i][task_mask]
            task_nll = (
                np.log(2 * np.pi * task_unc) / 2
                + (task_preds - task_targets) ** 2 / (2 * task_unc)
            )
            nll.append(task_nll.mean())
        return nll


class PlattCalibrator(UncertaintyCalibrator):
    """
    A calibration method for classification datasets based on the Platt scaling algorithm.
    As discussed in https://arxiv.org/abs/1706.04599.
    """

    @property
    def label(self):
        return f"{self.uncertainty_method}_platt_confidence"

    def raise_argument_errors(self):
        super().raise_argument_errors()
        if self.dataset_type != "classification":
            raise ValueError(
                "Platt scaling is only implemented for classification dataset types."
            )

    def calibrate(self):
        uncal_preds = np.array(
            self.calibration_predictor.get_uncal_preds()
        )  # shape(data, tasks)
        targets = np.array(self.calibration_data.targets())
        if self.calibration_data.is_atom_bond_targets:
            uncal_preds = [np.concatenate(x) for x in zip(*uncal_preds)]
            targets = [np.concatenate(x) for x in zip(*targets)]
        else:
            uncal_preds = np.array(list(zip(*uncal_preds)))
            targets = targets.astype(float)
            targets = np.array(list(zip(*targets)))
        mask = np.array(self.calibration_data.mask())
        self.num_tasks = len(mask)
        # If train class sizes are available, set Bayes corrected calibration targets
        if self.calibration_predictor.train_class_sizes is not None:
            class_size_correction = True
            train_class_sizes = np.sum(
                self.calibration_predictor.train_class_sizes, axis=0
            )  # shape(tasks, 2)
            negative_target = 1 / (train_class_sizes[:, 0] + 2)
            positive_target = (train_class_sizes[:, 1] + 1) / (
                train_class_sizes[:, 1] + 2
            )
            print(
                "Platt scaling for calibration uses Bayesian correction against training set overfitting, "
                "replacing calibration targets [0,1] with adjusted values."
            )
        else:
            class_size_correction = False
            print(
                "Class sizes used in training models unavailable in checkpoints before Chemprop v1.5.0. "
                "No Bayesian correction perfomed as part of class scaling."
            )

        platt_parameters = []
        for i in range(self.num_tasks):
            task_mask = mask[i]
            task_targets = targets[i][task_mask]
            task_preds = uncal_preds[i][task_mask]
            if class_size_correction:
                task_targets[task_targets == 0] = negative_target[i]
                task_targets[task_targets == 1] = positive_target[i]
                print(
                    f"Platt Bayesian correction for task {i} in calibration replacing [0,1] targets with {[negative_target[i], positive_target[i]]}"
                )

            def objective(parameters: np.ndarray):
                a = parameters[0]
                b = parameters[1]
                scaled_preds = expit(a * logit(task_preds) + b)
                nll = -1 * np.sum(
                    task_targets * np.log(scaled_preds)
                    + (1 - task_targets) * np.log(1 - scaled_preds)
                )
                return nll

            initial_guess = [1, 0]
            sol = fmin(objective, initial_guess)
            platt_parameters.append(sol)

        platt_parameters = np.array(platt_parameters)  # shape(task, 2)
        self.platt_a = platt_parameters[:, 0]
        self.platt_b = platt_parameters[:, 1]

    def apply_calibration(self, uncal_predictor: UncertaintyPredictor):
        uncal_preds = np.array(uncal_predictor.get_uncal_preds())  # shape(data, task)
        cal_preds = expit(
            np.expand_dims(self.platt_a, axis=0) * logit(uncal_preds)
            + np.expand_dims(self.platt_b, axis=0)
        )
        return uncal_preds.tolist(), cal_preds.tolist()

    def nll(self, preds: List[List[float]], unc: List[List[float]], targets: List[List[float]], mask: List[List[bool]]):
        targets = np.array(targets)
        unc = np.array(unc)
        mask = np.array(mask)
        if self.calibration_data.is_atom_bond_targets:
            targets = [np.concatenate(x) for x in zip(*targets)]
            unc = [np.concatenate(x) for x in zip(*unc)]
        else:
            targets = targets.astype(float)
            targets = np.array(list(zip(*targets)))
            unc = np.array(list(zip(*unc)))
        nll = []
        for i in range(self.num_tasks):
            task_mask = mask[i]
            task_targets = targets[i][task_mask]
            task_unc = unc[i][task_mask]

            likelihood = task_unc * task_targets + (1 - task_unc) * (1 - task_targets)
            task_nll = -1 * np.log(likelihood)
            nll.append(task_nll.mean())
        return nll


class IsotonicCalibrator(UncertaintyCalibrator):
    """
    A calibration method for classification datasets based on the isotonic regression algorithm.
    In effect, the method transforms incoming uncalibrated confidences using a histogram-like
    function where the range of each transforming bin and its magnitude is learned.
    As discussed in https://arxiv.org/abs/1706.04599.
    """

    @property
    def label(self):
        return f"{self.uncertainty_method}_isotonic_confidence"

    def raise_argument_errors(self):
        super().raise_argument_errors()
        if self.dataset_type != "classification":
            raise ValueError(
                "Isotonic Regression is only implemented for classification dataset types."
            )

    def calibrate(self):
        uncal_preds = np.array(
            self.calibration_predictor.get_uncal_preds()
        )  # shape(data, tasks)
        targets = np.array(self.calibration_data.targets())
        mask = np.array(self.calibration_data.mask())
        self.num_tasks = len(mask)

        if self.calibration_data.is_atom_bond_targets:
            uncal_preds = [np.concatenate(x) for x in zip(*uncal_preds)]
            targets = [np.concatenate(x) for x in zip(*targets)]
        else:
            uncal_preds = np.array(list(zip(*uncal_preds)))
            targets = targets.astype(float)
            targets = np.array(list(zip(*targets)))

        isotonic_models = []
        for i in range(self.num_tasks):
            task_mask = mask[i]
            task_targets = targets[i][task_mask]
            task_preds = uncal_preds[i][task_mask]

            isotonic_model = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip")
            isotonic_model.fit(task_preds, task_targets)
            isotonic_models.append(isotonic_model)

        self.isotonic_models = isotonic_models

    def apply_calibration(self, uncal_predictor: UncertaintyPredictor):
        uncal_preds = np.array(uncal_predictor.get_uncal_preds())  # shape(data, task)
        if self.calibration_data.is_atom_bond_targets:
            cal_preds = []
            uncal_preds_list = [np.concatenate(x) for x in zip(*uncal_preds)]
            for i, iso_model in enumerate(self.isotonic_models):
                task_preds = uncal_preds_list[i]
                task_cal = iso_model.predict(task_preds)
                transpose_cal_preds = [task_cal]
                cal_preds.append(np.transpose(transpose_cal_preds))
            return uncal_preds, cal_preds
        else:
            transpose_cal_preds = []
            for i, iso_model in enumerate(self.isotonic_models):
                task_preds = uncal_preds[:, i]
                task_cal = iso_model.predict(task_preds)
                transpose_cal_preds.append(task_cal)
            cal_preds = np.transpose(transpose_cal_preds)
            return uncal_preds.tolist(), cal_preds.tolist()

    def nll(self, preds: List[List[float]], unc: List[List[float]], targets: List[List[float]], mask: List[List[bool]]):
        targets = np.array(targets)
        mask = np.array(mask)
        unc = np.array(unc)
        if self.calibration_data.is_atom_bond_targets:
            targets = [np.concatenate(x) for x in zip(*targets)]
            unc = [np.concatenate(x) for x in zip(*unc)]
        else:
            targets = targets.astype(float)
            targets = np.array(list(zip(*targets)))
            unc = np.array(list(zip(*unc)))
        nll = []
        for i in range(self.num_tasks):
            task_mask = mask[i]
            task_targets = targets[i][task_mask]
            task_unc = unc[i][task_mask]

            likelihood = task_unc * task_targets + (1 - task_unc) * (1 - task_targets)
            task_nll = -1 * np.log(likelihood)
            nll.append(task_nll.mean())
        return nll


class IsotonicMulticlassCalibrator(UncertaintyCalibrator):
    """
    A multiclass method for classification datasets based on the isotonic regression algorithm.
    In effect, the method transforms incoming uncalibrated confidences using a histogram-like
    function where the range of each transforming bin and its magnitude is learned. Uses a
    one-against-all aggregation scheme for convertering between binary and multiclass classifiers.
    As discussed in https://arxiv.org/abs/1706.04599.
    """

    @property
    def label(self):
        return f"{self.uncertainty_method}_isotonic_confidence"

    def raise_argument_errors(self):
        super().raise_argument_errors()
        if self.dataset_type != "multiclass":
            raise ValueError(
                "Isotonic Multiclass Regression is only implemented for multiclass dataset types."
            )

    def calibrate(self):
        uncal_preds = np.array(
            self.calibration_predictor.get_uncal_preds()
        )  # shape(data, tasks, num_classes)
        targets = np.array(self.calibration_data.targets(), dtype=float)  # shape(data, tasks)
        mask = np.array(self.calibration_data.mask())
        self.num_tasks = len(mask)
        self.num_classes = uncal_preds.shape[2]

        isotonic_models = []
        for i in range(self.num_tasks):
            isotonic_models.append([])
            task_mask = mask[i]
            task_targets = targets[task_mask, i]  # shape(data)
            task_preds = uncal_preds[task_mask, i]
            for j in range(self.num_classes):
                class_preds = task_preds[:, j]  # shape(data)
                positive_class_targets = task_targets == j

                class_targets = np.ones_like(class_preds)
                class_targets[positive_class_targets] = 1
                class_targets[~positive_class_targets] = 0

                isotonic_model = IsotonicRegression(
                    y_min=0, y_max=1, out_of_bounds="clip"
                )
                isotonic_model.fit(class_preds, class_targets)
                isotonic_models[i].append(isotonic_model)

        self.isotonic_models = isotonic_models  # shape(tasks, classes)

    def apply_calibration(self, uncal_predictor: UncertaintyPredictor):
        uncal_preds = np.array(
            uncal_predictor.get_uncal_preds()
        )  # shape(data, task, class)
        transpose_cal_preds = []
        for i in range(self.num_tasks):
            transpose_cal_preds.append([])
            for j in range(self.num_classes):
                class_preds = uncal_preds[:, i, j]
                class_cal = self.isotonic_models[i][j].predict(class_preds)
                transpose_cal_preds[i].append(class_cal)  # shape (task, class, data)
        cal_preds = np.transpose(
            transpose_cal_preds, [2, 0, 1]
        )  # shape(data, task, class)
        cal_preds = cal_preds / np.sum(cal_preds, axis=2, keepdims=True)
        return uncal_preds.tolist(), cal_preds.tolist()

    def nll(self, preds: List[List[float]], unc: List[List[float]], targets: List[List[float]], mask: List[List[bool]]):
        targets = np.array(targets, dtype=int)  # shape(data, tasks)
        mask = np.array(mask)
        unc = np.array(unc)
        preds = np.array(preds)
        nll = []
        for i in range(targets.shape[1]):
            task_mask = mask[i]
            task_preds = unc[task_mask, i]
            task_targets = targets[task_mask, i]  # shape(data)
            bin_targets = np.zeros_like(preds[:, 0, :])  # shape(data, classes)
            bin_targets[np.arange(targets.shape[0]), task_targets] = 1
            task_likelihood = np.sum(bin_targets * task_preds, axis=1)
            task_nll = -1 * np.log(task_likelihood)
            nll.append(task_nll.mean())
        return nll


def build_uncertainty_calibrator(
    calibration_method: str,
    uncertainty_method: str,
    regression_calibrator_metric: str,
    interval_percentile: int,
    calibration_data: MoleculeDataset,
    calibration_data_loader: MoleculeDataLoader,
    models: Iterator[MoleculeModel],
    scalers: Iterator[StandardScaler],
    num_models: int,
    dataset_type: str,
    loss_function: str,
    uncertainty_dropout_p: float,
    dropout_sampling_size: int,
    spectra_phase_mask: List[List[bool]],
) -> UncertaintyCalibrator:
    """
    Function that chooses the subclass of :class: `UncertaintyCalibrator`
    based on the provided arguments and returns that class.
    """
    if calibration_method is None:
        if dataset_type == "regression":
            if regression_calibrator_metric == "stdev":
                calibration_method = "zscaling"
            else:
                calibration_method = "zelikman_interval"
        if dataset_type in ["classification", "multiclass"]:
            calibration_method == "isotonic"

    supported_calibrators = {
        "zscaling": ZScalingCalibrator,
        "tscaling": TScalingCalibrator,
        "zelikman_interval": ZelikmanCalibrator,
        "mve_weighting": MVEWeightingCalibrator,
        "platt": PlattCalibrator,
        "isotonic": IsotonicCalibrator
        if dataset_type == "classification"
        else IsotonicMulticlassCalibrator,
    }

    calibrator_class = supported_calibrators.get(calibration_method, None)

    if calibrator_class is None:
        raise NotImplementedError(
            f"Calibrator type {calibration_method} is not currently supported. Avalable options are: {list(supported_calibrators.keys())}"
        )
    else:
        calibrator = calibrator_class(
            uncertainty_method=uncertainty_method,
            regression_calibrator_metric=regression_calibrator_metric,
            interval_percentile=interval_percentile,
            calibration_data=calibration_data,
            calibration_data_loader=calibration_data_loader,
            models=models,
            scalers=scalers,
            num_models=num_models,
            dataset_type=dataset_type,
            loss_function=loss_function,
            uncertainty_dropout_p=uncertainty_dropout_p,
            dropout_sampling_size=dropout_sampling_size,
            spectra_phase_mask=spectra_phase_mask,
        )
    return calibrator
