from abc import ABC, abstractmethod
from typing import Iterator, List

import numpy as np
from chemprop.data.data import MoleculeDataLoader
from scipy.special import erfinv, softmax, logit, expit
from scipy.optimize import least_squares, fmin
from scipy.stats import t
from sklearn.isotonic import IsotonicRegression

from chemprop.data import MoleculeDataset, StandardScaler
from chemprop.models import MoleculeModel
from chemprop.uncertainty.uncertainty_predictor import build_uncertainty_predictor, UncertaintyPredictor


class UncertaintyCalibrator(ABC):
    """
    Uncertainty calibrator class. Subclasses for each uncertainty calibration
    method. Subclasses should override the calibrate and apply functions for
    implemented metrics.
    """

    def __init__(
        self,
        uncertainty_method: str,
        calibration_method: str,
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
        alpha: float,
        dropout_sampling_size: int,
        spectra_phase_mask: List[List[bool]],
    ):
        self.calibration_data = calibration_data
        self.calibration_data_loader = calibration_data_loader
        self.regression_calibrator_metric = regression_calibrator_metric
        self.interval_percentile = interval_percentile
        self.dataset_type = dataset_type
        self.uncertainty_method = uncertainty_method
        self.calibration_method = calibration_method
        self.loss_function = loss_function
        self.num_models = num_models
        self.alpha = alpha

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
            alpha=alpha,
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
        if self.uncertainty_method in ['ensemble', 'dropout'] and self.dataset_type in ['classification', 'multiclass']:
            raise NotImplementedError(
                'Though ensemble and dropout uncertainty methods are available for classification \
                    multiclass dataset types, their outputs are not confidences and are not \
                    compatible with any implemented calibration methods for classification.'
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

    def nll(
        self,
        preds: List[List[float]],
        unc: List[List[float]],
        targets: List[List[float]],
    ):
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
            label = f"{self.uncertainty_method}_zscaling_{self.interval_percentile}interval"
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
        errors = uncal_preds - targets
        zscore_preds = errors / np.sqrt(uncal_vars)

        def objective(scaler_values: np.ndarray):
            scaled_vars = uncal_vars * np.expand_dims(scaler_values, axis=0) ** 2
            nll = np.log(2 * np.pi * scaled_vars) / 2 \
                + (errors) ** 2 / (2 * scaled_vars)
            nll = np.sum(nll, axis=0)
            return nll

        initial_guess = np.std(zscore_preds, axis=0, keepdims=False)
        sol = least_squares(objective, initial_guess)
        stdev_scaling = sol.x
        if self.regression_calibrator_metric == "stdev":
            self.scaling = stdev_scaling
        else:  # interval
            interval_scaling = (
                stdev_scaling * erfinv(self.interval_percentile / 100) * np.sqrt(2)
            )
            self.scaling = interval_scaling

    def apply_calibration(self, uncal_predictor: UncertaintyPredictor):
        uncal_preds = np.array(uncal_predictor.get_uncal_preds())
        uncal_vars = np.array(uncal_predictor.get_uncal_vars())
        cal_stdev = np.sqrt(uncal_vars) * np.expand_dims(self.scaling, axis=0)
        return uncal_preds.tolist(), cal_stdev.tolist()

    def nll(
        self,
        preds: List[List[float]],
        unc: List[List[float]],
        targets: List[List[float]],
    ):
        unc_var = np.square(unc)
        preds = np.array(preds)
        targets = np.array(targets)
        return (
            np.log(2 * np.pi * unc_var) / 2
            + (preds - targets) ** 2 / (2 * unc_var)
        )


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
            label = f"{self.uncertainty_method}_tscaling_{self.interval_percentile}interval"
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
        std_error_of_mean = np.sqrt(
            uncal_vars / (self.num_models - 1)
        )  # reduced for number of samples and include Bessel's correction
        targets = np.array(self.calibration_data.targets())
        errors = uncal_preds - targets
        tscore_preds = errors / std_error_of_mean

        def objective(scaler_values: np.ndarray):
            scaled_std = std_error_of_mean * np.expand_dims(scaler_values, axis=0)
            likelihood = t.pdf(
                x=errors, df=self.num_models - 1, scale=scaled_std
            )  # scipy t distribution pdf
            nll = -1 * np.sum(np.log(likelihood), axis=0)
            return nll

        initial_guess = np.std(tscore_preds, axis=0, keepdims=False)
        sol = least_squares(objective, initial_guess)
        stdev_scaling = sol.x
        if self.regression_calibrator_metric == "stdev":
            self.scaling = stdev_scaling
        else:  # interval
            interval_scaling = stdev_scaling * t.ppf(
                (self.interval_percentile / 100 + 1) / 2, df=self.num_models - 1
            )
            self.scaling = interval_scaling

    def apply_calibration(self, uncal_predictor: UncertaintyPredictor):
        uncal_preds = np.array(uncal_predictor.get_uncal_preds())
        uncal_vars = np.array(uncal_predictor.get_uncal_vars())
        cal_stdev = np.sqrt(uncal_vars / (self.num_models - 1)) * np.expand_dims(
            self.scaling, axis=0
        )
        return uncal_preds.tolist(), cal_stdev.tolist()

    def nll(
        self,
        preds: List[List[float]],
        unc: List[List[float]],
        targets: List[List[float]],
    ):
        return -1 * t.logpdf(
            x=np.array(preds) - np.array(targets), scale=unc, df=self.num_models - 1
        )


class ZelikmanCalibrator(UncertaintyCalibrator):
    """
    A calibrator for regression datasets that does not depend on a particular probability
    function form. Designed to be used with interval output. Uses the "CRUDE" method as
    described in https://arxiv.org/abs/2005.12496. As implemented here, the interval
    bounds are constrained to be symmetrical, though this is not required in the source method.
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
        abs_zscore_preds = np.abs(uncal_preds - targets) / np.sqrt(uncal_vars)
        if self.regression_calibrator_metric == "interval":
            interval_scaling = np.percentile(
                abs_zscore_preds, self.interval_percentile, axis=0
            )
            self.scaling = interval_scaling
        else:
            symmetric_z = np.concatenate(
                [abs_zscore_preds, -1 * abs_zscore_preds], axis=0
            )
            std_scaling = np.std(symmetric_z, axis=0)
            self.scaling = std_scaling
        # histogram parameters for nll calculation
        self.num_tasks = targets.shape[1]
        self.histogram_parameters = []
        for i in range(self.num_tasks):
            h_params = np.histogram(
            abs_zscore_preds[:, i], bins='auto', density=True
            )
            self.histogram_parameters.append(h_params)

    def apply_calibration(self, uncal_predictor: UncertaintyPredictor):
        uncal_preds = np.array(uncal_predictor.get_uncal_preds())
        uncal_vars = np.array(uncal_predictor.get_uncal_vars())
        cal_stdev = np.sqrt(uncal_vars) * np.expand_dims(self.scaling, axis=0)
        return uncal_preds.tolist(), cal_stdev.tolist()

    def nll(
        self,
        preds: List[List[float]],
        unc: List[List[float]],
        targets: List[List[float]],
    ):
    # uses a histogram as probability density function
    # assumes symmetrical probability distribution
        preds = np.array(preds)
        unc = np.array(unc)
        targets = np.array(targets)
        nll = np.zeros_like(preds)
        for i in range(self.num_tasks):
            task_preds = preds[:, i]
            task_targets = targets[:, i]
            task_stdev = unc[:, i] / self.scaling[i]
            task_abs_z = np.abs(task_preds - task_targets) / task_stdev
            bin_edges = self.histogram_parameters[i][1]
            bin_magnitudes = self.histogram_parameters[i][0]
            bin_magnitudes = np.insert(
                bin_magnitudes, [0, len(bin_magnitudes)], 0
            )
            pred_bins = np.searchsorted(bin_edges, task_abs_z)
            # magnitude adjusted by stdev scale of the distribution and symmetry assumption
            task_likelihood = bin_magnitudes[pred_bins] / task_stdev / 2
            nll[:, i] = -1 * np.log(task_likelihood)
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
                "MVE Weighting is only compatible with regression datasets."
            )
        if self.loss_function not in ["mve", "evidential"]:
            raise ValueError(
                "MVE Weighting calibration can only be carried out with MVE or Evidential loss function models."
            )

    def calibrate(self):
        uncal_preds = np.array(
            self.calibration_predictor.get_uncal_preds()
        )  # shape(data, tasks)
        individual_vars = np.array(
            self.calibration_predictor.get_individual_vars()
        )  # shape(models, data, tasks)
        targets = np.array(self.calibration_data.targets())
        errors = uncal_preds - targets

        def objective(scaler_values: np.ndarray):
            scaler_values = np.reshape(softmax(scaler_values), [-1, 1, 1])
            scaled_vars = np.sum(
                individual_vars * scaler_values, axis=0, keepdims=False
            )
            nll = np.log(2 * np.pi * scaled_vars) / 2 + (errors) ** 2 / (
                2 * scaled_vars
            )
            nll = np.sum(nll, axis=0)
            return nll

        initial_guess = np.ones_like(self.num_models)
        sol = fmin(objective, initial_guess)
        self.var_weighting = softmax(sol)
        if self.regression_calibrator_metric == "stdev":
            self.scaling = 1
        else:  # interval
            self.scaling = erfinv(self.interval_percentile / 100) * np.sqrt(2)

    def apply_calibration(self, uncal_predictor: UncertaintyPredictor):
        uncal_preds = np.array(uncal_predictor.get_uncal_preds())
        uncal_individual_vars = np.array(uncal_predictor.get_individual_vars())
        weighted_vars = np.sum(
            uncal_individual_vars * np.reshape(self.var_weighting, [-1, 1, 1]),
            axis=0,
            keepdims=False,
        )
        weighted_stdev = np.sqrt(weighted_vars) * self.scaling
        return uncal_preds.tolist(), weighted_stdev.tolist()

    def nll(
        self,
        preds: List[List[float]],
        unc: List[List[float]],
        targets: List[List[float]],
    ):
        preds = np.array(preds)
        targets = np.array(targets)
        unc_var = np.square(unc)
        return np.log(2 * np.pi * unc_var) / 2 \
            + (preds - targets) ** 2 / (2 * unc_var)


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
        num_tasks = targets.shape[1]
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
                f"replacing calibration targets [0,1] with adjusted values."
            )
        else:
            class_size_correction = False
            print(
                "Class sizes used in training models unavailable in checkpoints before Chemprop v1.5.0. "
                "No Bayesian correction perfomed as part of class scaling."
            )

        platt_parameters = []
        for i in range(num_tasks):
            task_targets = targets[:, i]
            task_preds = uncal_preds[:, i]
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
        num_tasks = targets.shape[1]

        isotonic_models = []
        for i in range(num_tasks):
            task_targets = targets[:, i]
            task_preds = uncal_preds[:, i]

            isotonic_model = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip")
            isotonic_model.fit(task_preds, task_targets)
            isotonic_models.append(isotonic_model)

        self.isotonic_models = isotonic_models

    def apply_calibration(self, uncal_predictor: UncertaintyPredictor):
        uncal_preds = np.array(uncal_predictor.get_uncal_preds())  # shape(data, task)
        transpose_cal_preds = []
        for i, iso_model in enumerate(self.isotonic_models):
            task_preds = uncal_preds[:, i]
            task_cal = iso_model.predict(task_preds)
            transpose_cal_preds.append(task_cal)
        cal_preds = np.transpose(transpose_cal_preds)
        return uncal_preds.tolist(), cal_preds.tolist()

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
        targets = np.array(self.calibration_data.targets())  # shape(data, tasks)
        self.num_tasks = targets.shape[1]
        self.num_classes = uncal_preds.shape[2]

        isotonic_models = []
        for i in range(self.num_tasks):
            isotonic_models.append([])
            task_targets = targets[:, i]  # shape(data)
            for j in range(self.num_classes):
                class_preds = uncal_preds[:, i, j]  # shape(data)
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


class ConformalMulticlassCalibrator(UncertaintyCalibrator):
    """
    Conformal Calibrator. Outputs binary values for whether each class should be included in the
    conformal set for each task.
    """
    @property
    def label(self):
        return "conformal"

    def raise_argument_errors(self):
        super().raise_argument_errors()
        if self.dataset_type != "multiclass":
            raise ValueError(
                "Conformal is only implemented for multiclass dataset types."
            )

    @staticmethod
    def nonconformity_scores(uncal_preds):
        """Fixed per class. Example is for basic conformal.                                                                                                                                                                                                                              
                                                                                                                                                                                                                                                                              
        Args:                                                                                                                                                                                                                                                                     
            preds: [num_examples, num_tasks, num_classes]                                                                                                                                                                                                                         
                                                                                                                                                                                                                                                                              
        Returns:                                                                                                                                                                                                                                                                  
            scores: [num_examples, num_tasks, num_classes]                                                                                                                                                                                                                        
        """
        return -uncal_preds

    def calibrate(self):
        uncal_preds = np.array(self.calibration_predictor.get_uncal_preds())  # shape(data, tasks, num_classes)
        targets = np.array(self.calibration_data.targets(), dtype=int)  # shape(data, tasks)
        mask = ~np.isnan(targets) # keep track of invalid data shape(data, tasks)
        targets = np.nan_to_num(targets, copy=True, nan=0.0, posinf=None, neginf=None)
        num_data = targets.shape[0]
        self.num_tasks = targets.shape[1]
        self.num_classes = uncal_preds.shape[2]

        all_scores = self.nonconformity_scores(uncal_preds)
        self.qhats = []

        for task_id in range(self.num_tasks):
            # Extract scores for true targets for current task
            task_scores = np.take_along_axis(all_scores[:, task_id], np.expand_dims(targets[:, task_id], -1), axis=1).squeeze(axis=1)

            # Only keep valid data
            task_scores = task_scores[mask[:, task_id]] # shape(valid_data)
            
            # Pad scores with +Inf and sort
            task_scores = np.append(task_scores, np.Inf)
            task_scores = np.sort(task_scores)

            #Compute empirical quantile
            qhat = np.quantile(task_scores, 1 - self.alpha / self.num_tasks)
            self.qhats.append(qhat)

    def apply_calibration(self, uncal_predictor: UncertaintyPredictor):
        uncal_preds = np.array(uncal_predictor.get_uncal_preds())  # shape(data, task)
        cal_preds = np.zeros_like(uncal_preds, dtype=int)
        all_scores = self.nonconformity_scores(uncal_preds)
        for task_id, qhat in enumerate(self.qhats):
            cal_preds[:, task_id, :] = (all_scores[:, task_id, :] <= qhat).astype(int)
        return uncal_preds.tolist(), cal_preds.tolist()


class ConformalAdaptiveMulticlassCalibrator(ConformalMulticlassCalibrator):
    """
    Adaptive Conformal Calibrator. Outputs binary values for whether each class should be
    included in the conformal set for each task.
    """
    @property
    def label(self):
        return "conformal_adaptive"

    def raise_argument_errors(self):
        super().raise_argument_errors()
        if self.dataset_type != "multiclass":
            raise ValueError(
                "Conformal Adaptive is only implemented for multiclass dataset types."
            )

    @staticmethod
    def nonconformity_scores(uncal_preds):
        """Fixed per class. Example is for adaptive.                                                                                                                                                                                                                              
                                                                                                                                                                                                                                                                              
        Args:                                                                                                                                                                                                                                                                     
            preds: [num_examples, num_tasks, num_classes]                                                                                                                                                                                                                         
                                                                                                                                                                                                                                                                              
        Returns:                                                                                                                                                                                                                                                                  
            scores: [num_examples, num_tasks, num_classes]                                                                                                                                                                                                                        
        """
        # Sort indices from high to low
        sort_inds = np.argsort(-uncal_preds, axis=2)

        # Sort and calculate scores
        sorted_preds = np.take_along_axis(uncal_preds, sort_inds, axis=2)
        sorted_scores = sorted_preds.cumsum(axis=2)

        # Unsort and return
        unsort_inds = np.argsort(sort_inds, axis=2)
        unsorted_scores = np.take_along_axis(sorted_scores, unsort_inds, axis=2)
        return unsorted_scores


class ConformalMultilabelCalibrator(UncertaintyCalibrator):
    """
    Conformal Calibrator.
    """
    @property
    def label(self):
        return f"conformal_multilabel"

    def raise_argument_errors(self):
        super().raise_argument_errors()
        if self.dataset_type != "classification":
            raise ValueError(
                "Conformal is only implemented for classification dataset types."
            )

    @staticmethod
    def nonconformity_scores(uncal_preds):
        """Fixed per class. Example is for multilabel conformal.                                                                                                                                                                                                                              
                                                                                                                                                                                                                                                                              
        Args:                                                                                                                                                                                                                                                                     
            preds: [num_examples, num_tasks, num_classes]                                                                                                                                                                                                                         
                                                                                                                                                                                                                                                                              
        Returns:                                                                                                                                                                                                                                                                  
            scores: [num_examples, num_tasks, num_classes]                                                                                                                                                                                                                        
        """
        return -uncal_preds

    def calibrate(self):
        uncal_preds = np.array(self.calibration_predictor.get_uncal_preds())  # shape(data, tasks, num_classes)
        targets = np.array(self.calibration_data.targets(), dtype=bool)  # shape(data, tasks)
        (self.num_data, self.num_tasks) = targets.shape

        # Calculate calibration scores for in set
        has_zeros = np.sum(targets==0, axis=1) > 0
        inds_zeros = targets[has_zeros] == 0
        scores_in = self.nonconformity_scores(uncal_preds[has_zeros])
        masked_scores_in = scores_in * inds_zeros + np.nan_to_num(np.Inf * (1 - inds_zeros).astype(float), copy=True, nan=0.0, posinf=None, neginf=None)
        calibration_scores_in = np.min(masked_scores_in, axis=1)

        # Calculate calibration scores for out set
        has_ones = np.sum(targets==1, axis=1) > 0
        inds_ones = targets[has_ones] == 1
        scores_out = self.nonconformity_scores(uncal_preds[has_ones])
        masked_scores_out = scores_out * inds_ones + np.nan_to_num(-np.Inf * (1 - inds_ones).astype(float), copy=True, nan=0.0, posinf=None, neginf=None)
        calibration_scores_out = np.max(masked_scores_out, axis=1)
        
        self.tout = np.quantile(calibration_scores_out, 1 - self.alpha / 2)
        self.tin = np.quantile(calibration_scores_in, self.alpha / 2)

    def apply_calibration(self, uncal_predictor: UncertaintyPredictor):
        uncal_preds = np.array(uncal_predictor.get_uncal_preds())  # shape(data, task)
        scores = self.nonconformity_scores(uncal_preds)

        cal_preds_in = (scores <= self.tin).astype(int)
        cal_preds_out = (scores <= self.tout).astype(int)
        cal_preds = np.concatenate((cal_preds_in, cal_preds_out), axis=1)

        return uncal_preds.tolist(), cal_preds.tolist()


class ConformalQuantileRegressionCalibrator(UncertaintyCalibrator):
    """
    Conformal Quantile Regression Calibrator.
    """
    @property
    def label(self):
        return "conformal_quantile_regression"

    def raise_argument_errors(self):
        super().raise_argument_errors()
        if self.dataset_type != "regression":
            raise ValueError(
                "Conformal Quantile Regression is only implemented for regression dataset types."
            )

    @staticmethod
    def get_preds(predictor: UncertaintyPredictor):
        return np.array(predictor.get_uncal_preds())

    def calibrate(self):
        uncal_preds = self.get_preds(self.calibration_predictor) # shape(data, 2 * tasks)
        targets = np.array(self.calibration_data.targets(), dtype=float)  # shape(data, tasks)
        targets = np.nan_to_num(targets, copy=True, nan=0.0, posinf=None, neginf=None)
     
        self.num_tasks = uncal_preds.shape[1] // 2
        self.qhats = []
        
        for task_id in range(self.num_tasks):
            targets_task_id = targets[:, task_id]
            uncal_preds_lower = uncal_preds[:, task_id]
            uncal_preds_upper = uncal_preds[:, task_id + self.num_tasks]

            calibration_scores = np.maximum(uncal_preds_lower - targets_task_id, targets_task_id - uncal_preds_upper)
            calibration_scores = np.append(calibration_scores, np.Inf)
            calibration_scores = np.sort(np.absolute(calibration_scores))
            self.qhats.append(np.quantile(calibration_scores, 1 - self.alpha / self.num_tasks))

    def apply_calibration(self, uncal_predictor: UncertaintyPredictor):
        uncal_preds = self.get_preds(uncal_predictor)  # shape(data, task)
        num_data = uncal_preds.shape[0]
        intervals = np.zeros((2*self.num_tasks, num_data), dtype=float)

        for task_id in range(self.num_tasks):
            uncal_preds_lower = uncal_preds[:, task_id]
            uncal_preds_upper = uncal_preds[:, task_id + self.num_tasks]

            intervals[task_id] = uncal_preds_lower - self.qhats[task_id]
            intervals[task_id + self.num_tasks] = uncal_preds_upper + self.qhats[task_id]
        
        intervals = np.transpose(intervals)
        return uncal_preds.tolist(), intervals.tolist()


class ConformalRegressionCalibrator(ConformalQuantileRegressionCalibrator):
    """
    Conformal Regression Calibrator.
    """
    @property
    def label(self):
        return f"{self.uncertainty_method}_conformal_regression"

    def raise_argument_errors(self):
        super().raise_argument_errors()
        if self.dataset_type != "regression":
            raise ValueError(
                "Conformal Regression is only implemented for regression dataset types."
            )

    @staticmethod
    def get_preds(predictor: UncertaintyPredictor):
        preds = np.array(predictor.get_uncal_preds())

        # duplicate preds to simulate upper and lower quantile
        preds_duplicated = np.concatenate((preds, preds), axis=1)
        return preds_duplicated


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
    alpha: float,
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
        "conformal": ConformalMultilabelCalibrator if dataset_type == "classification" else ConformalMulticlassCalibrator,
        "conformal_adaptive": ConformalAdaptiveMulticlassCalibrator,
        "conformal_regression": ConformalRegressionCalibrator,
        "conformal_quantile_regression": ConformalQuantileRegressionCalibrator,
        "isotonic": IsotonicCalibrator if dataset_type == "classification" else IsotonicMulticlassCalibrator,
    }

    calibrator_class = supported_calibrators.get(calibration_method, None)

    if calibrator_class is None:
        raise NotImplementedError(
            f"Calibrator type {calibration_method} is not currently supported. Avalable options are: {list(supported_calibrators.keys())}"
        )
    else:
        calibrator = calibrator_class(
            uncertainty_method=uncertainty_method,
            calibration_method=calibration_method,
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
            alpha=alpha,
            dropout_sampling_size=dropout_sampling_size,
            spectra_phase_mask=spectra_phase_mask,
        )
    return calibrator
