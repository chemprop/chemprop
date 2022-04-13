from typing import Iterator

import numpy as np
from scipy.special import erfinv, erf, softmax, logit, expit
from scipy.optimize import least_squares, fmin
from scipy.stats import norm, t
from sklearn.isotonic import IsotonicRegression

from chemprop.data import MoleculeDataset, StandardScaler
from chemprop.models import MoleculeModel
from .uncertainty_predictor import uncertainty_predictor_builder, UncertaintyPredictor
from .utils import calibration_normal_auc

class UncertaintyCalibrator:
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
        models: Iterator[MoleculeModel],
        scalers: Iterator[StandardScaler],
        dataset_type: str,
        loss_function: str,
        batch_size: int,
        num_workers: int,
        dropout_sampling_size: int,
    ):
        self.calibration_data = calibration_data
        self.regression_calibrator_metric = regression_calibrator_metric
        self.interval_percentile = interval_percentile
        self.dataset_type = dataset_type
        self.loss_function = loss_function
        self.num_models = len(models)

        self.raise_argument_errors()

        self.calibration_predictor = uncertainty_predictor_builder(
            test_data=calibration_data,
            models=models,
            scalers=scalers,
            dataset_type=dataset_type,
            batch_size=batch_size,
            num_workers=num_workers,
            loss_function=loss_function,
            uncertainty_method=uncertainty_method,
            dropout_sampling_size=dropout_sampling_size,
        )

        self.calibrate()
    
    def raise_argument_errors(self):
        """
        Raise errors for incompatibilities between dataset type and uncertainty method, or similar.
        """
        pass

    def calibrate(self):
        """
        Fit calibration method for the calibration data.
        """
        pass

    def apply_calibration(self, uncal_predictor: UncertaintyPredictor):
        """
        Take in predictions and uncertainty parameters from a model and apply the calibration method using fitted parameters.
        """
        pass


class ZScalingCalibrator(UncertaintyCalibrator):
    def __init__(
        self,
        uncertainty_method: str,
        interval_percentile: int,
        regression_calibrator_metric: str,
        calibration_data: MoleculeDataset,
        models: Iterator[MoleculeModel],
        scalers: Iterator[StandardScaler],
        dataset_type: str,
        loss_function: str,
        batch_size: int,
        num_workers: int,
        dropout_sampling_size: int,
    ):
        super().__init__(
            uncertainty_method=uncertainty_method,
            interval_percentile=interval_percentile,
            regression_calibrator_metric=regression_calibrator_metric,
            calibration_data=calibration_data,
            models=models,
            scalers=scalers,
            dataset_type=dataset_type,
            loss_function=loss_function,
            batch_size=batch_size,
            num_workers=num_workers,
            dropout_sampling_size=dropout_sampling_size,
        )
        if self.regression_calibrator_metric == 'stdev':
            self.label = f'{uncertainty_method}_zscaling_stdev'
        else: # interval
            self.label = f'{uncertainty_method}_zscaling_{interval_percentile}interval'

    def raise_argument_errors(self):
        super().raise_argument_errors()
        if self.dataset_type != 'regression':
            raise ValueError('Z Score Scaling is only compatible with regression datasets.')

    def calibrate(self):
        uncal_preds = np.array(self.calibration_predictor.get_uncal_preds()) # shape(data, tasks)
        uncal_vars = np.array(self.calibration_predictor.get_uncal_vars())
        targets = np.array(self.calibration_data.targets())
        errors = uncal_preds - targets
        zscore_preds = errors / np.sqrt(uncal_vars)

        def objective(scaler_values: np.ndarray):
            scaled_vars = uncal_vars * np.expand_dims(scaled_vars, axis=0) ** 2
            nll = np.log(2 * np.pi * scaled_vars) / 2 + (errors) ** 2 / (2 * scaled_vars)
            nll = np.sum(nll, axis=0)
            return nll

        initial_guess = np.std(zscore_preds, axis=0, keepdims=False)
        sol = least_squares(objective, initial_guess)
        stdev_scaling = sol.x
        if self.regression_calibrator_metric == 'stdev':
            self.scaling = stdev_scaling
        else: # interval
            interval_scaling = stdev_scaling * erfinv(self.interval_percentile/100) * np.sqrt(2)
            self.scaling = interval_scaling

    def apply_calibration(self, uncal_predictor: UncertaintyPredictor):
        uncal_preds = uncal_predictor.get_uncal_preds()
        uncal_vars = uncal_predictor.get_uncal_vars()
        cal_stdev = np.sqrt(uncal_vars) * np.expand_dims(self.scaling, axis=0)
        return uncal_preds, cal_stdev.tolist()


class TScalingCalibrator(UncertaintyCalibrator):
    def __init__(
        self,
        uncertainty_method: str,
        interval_percentile: int,
        regression_calibrator_metric: str,
        calibration_data: MoleculeDataset,
        models: Iterator[MoleculeModel],
        scalers: Iterator[StandardScaler],
        dataset_type: str,
        loss_function: str,
        batch_size: int,
        num_workers: int,
        dropout_sampling_size: int,
    ):
        super().__init__(
            uncertainty_method=uncertainty_method,
            interval_percentile=interval_percentile,
            regression_calibrator_metric=regression_calibrator_metric,
            calibration_data=calibration_data,
            models=models,
            scalers=scalers,
            dataset_type=dataset_type,
            loss_function=loss_function,
            batch_size=batch_size,
            num_workers=num_workers,
            dropout_sampling_size=dropout_sampling_size,
        )
        if self.regression_calibrator_metric == 'stdev':
            self.label = f'{uncertainty_method}_tscaling_stdev'
        else: # interval
            self.label = f'{uncertainty_method}_tscaling_{interval_percentile}interval'

    def raise_argument_errors(self):
        super().raise_argument_errors()
        if self.dataset_type != 'regression':
            raise ValueError('T Score Scaling is only compatible with regression datasets.')

    def calibrate(self):
        uncal_preds = np.array(self.calibration_predictor.get_uncal_preds()) # shape(data, tasks)
        uncal_vars = np.array(self.calibration_predictor.get_uncal_vars())
        std_error_of_mean = np.sqrt(uncal_vars / (self.num_models - 1) ) # reduced for number of samples and include Bessel's correction
        targets = np.array(self.calibration_data.targets())
        errors = uncal_preds - targets
        tscore_preds = errors / std_error_of_mean

        def objective(scaler_values: np.ndarray):
            scaled_std = std_error_of_mean * np.expand_dims(scaler_values, axis=0)
            likelihood = t.pdf(x=errors, df=self.num_models - 1, scale = scaled_std) # scipy t distribution pdf
            nll = -1 * np.sum(np.log(likelihood), axis=0)
            return nll

        initial_guess = np.std(tscore_preds, axis=0, keepdims=False)
        sol = least_squares(objective, initial_guess)
        stdev_scaling = sol.x
        if self.regression_calibrator_metric == 'stdev':
            self.scaling = stdev_scaling
        else: # interval
            interval_scaling = stdev_scaling * t.ppf((self.interval_percentile/100 +1) / 2, df = self.num_models - 1)
            self.scaling = interval_scaling

    def apply_calibration(self, uncal_predictor: UncertaintyPredictor):
        uncal_preds = uncal_predictor.get_uncal_preds()
        uncal_vars = uncal_predictor.get_uncal_vars()
        cal_stdev = np.sqrt(uncal_vars / (self.num_models -1 )) * np.expand_dims(self.scaling, axis=0)
        return uncal_preds, cal_stdev.tolist()


class ZelikmanCalibrator(UncertaintyCalibrator):
    def __init__(
        self,
        uncertainty_method: str,
        interval_percentile: int,
        regression_calibrator_metric: str,
        calibration_data: MoleculeDataset,
        models: Iterator[MoleculeModel],
        scalers: Iterator[StandardScaler],
        dataset_type: str,
        loss_function: str,
        batch_size: int,
        num_workers: int,
        dropout_sampling_size: int,
    ):
        super().__init__(
            uncertainty_method=uncertainty_method,
            interval_percentile=interval_percentile,
            regression_calibrator_metric=regression_calibrator_metric,
            calibration_data=calibration_data,
            models=models,
            scalers=scalers,
            dataset_type=dataset_type,
            loss_function=loss_function,
            batch_size=batch_size,
            num_workers=num_workers,
            dropout_sampling_size=dropout_sampling_size,
        )
        self.label = f'{uncertainty_method}_zelikman_{interval_percentile}interval'

    def raise_argument_errors(self):
        super().raise_argument_errors()
        if self.dataset_type != 'regression':
            raise ValueError('Crude Scaling is only compatible with regression datasets.')

    def calibrate(self):
        uncal_preds = np.array(self.calibration_predictor.get_uncal_preds()) # shape(data, tasks)
        uncal_vars = np.array(self.calibration_predictor.get_uncal_vars())
        targets = np.array(self.calibration_data.targets())
        abs_zscore_preds = np.abs(uncal_preds - targets) / np.sqrt(uncal_vars)
        interval_scaling = np.percentile(abs_zscore_preds, self.interval_percentile, axis=0, keepdims=True)
        self.scaling = interval_scaling

    def apply_calibration(self, uncal_predictor: UncertaintyPredictor):
        uncal_preds = uncal_predictor.get_uncal_preds()
        uncal_vars = uncal_predictor.get_uncal_vars()
        cal_stdev = np.sqrt(uncal_vars) * self.scaling
        return uncal_preds, cal_stdev.tolist()


class MVEWeightingCalibrator(UncertaintyCalibrator):
    def __init__(
        self,
        uncertainty_method: str,
        interval_percentile: int,
        regression_calibrator_metric: str,
        calibration_data: MoleculeDataset,
        models: Iterator[MoleculeModel],
        scalers: Iterator[StandardScaler],
        dataset_type: str,
        loss_function: str,
        batch_size: int,
        num_workers: int,
        dropout_sampling_size: int,
    ):
        super().__init__(
            uncertainty_method=uncertainty_method,
            interval_percentile=interval_percentile,
            regression_calibrator_metric=regression_calibrator_metric,
            calibration_data=calibration_data,
            models=models,
            scalers=scalers,
            dataset_type=dataset_type,
            loss_function=loss_function,
            batch_size=batch_size,
            num_workers=num_workers,
            dropout_sampling_size=dropout_sampling_size,
        )
        if self.regression_calibrator_metric == 'stdev':
            self.label = f'{uncertainty_method}_mve_weighting_stdev'
        else: # interval
            self.label = f'{uncertainty_method}_mve_weighting_{interval_percentile}interval'

    def raise_argument_errors(self):
        super().raise_argument_errors()
        if self.dataset_type != 'regression':
            raise ValueError('MVE Weighting is only compatible with regression datasets.')
        if self.loss_function not in ['mve', 'evidential_epistemic', 'evidential_aleatoric', 'evidential_total']:
            raise ValueError('MVE Weighting calibration can only be carried out with MVE or Evidential loss function models.')

    def calibrate(self):
        uncal_preds = np.array(self.calibration_predictor.get_uncal_preds()) # shape(data, tasks)
        individual_vars = np.array(self.calibration_predictor.get_individual_vars()) #shape(models, data, tasks)
        targets = np.array(self.calibration_data.targets())
        errors = uncal_preds - targets

        def objective(scaler_values: np.ndarray):
            scaler_values = np.reshape(softmax(scaler_values), [-1, 1, 1])
            scaled_vars = np.sum(individual_vars * scaler_values, axis=0, keepdims=False)
            nll = np.log(2 * np.pi * scaled_vars) / 2 + (errors) ** 2 / (2 * scaled_vars)
            nll = np.sum(nll, axis=0)
            return nll

        initial_guess = np.ones_like(self.num_models)
        sol = fmin(objective, initial_guess)
        self.var_weighting = softmax(sol)
        if self.regression_calibrator_metric == 'stdev':
            self.scaling = 1
        else: # interval
            self.scaling = erfinv(self.interval_percentile/100) * np.sqrt(2)

    def apply_calibration(self, uncal_predictor: UncertaintyPredictor):
        uncal_preds = uncal_predictor.get_uncal_preds()
        uncal_individual_vars = uncal_predictor.get_individual_vars()
        weighted_vars = np.sum(uncal_individual_vars * np.reshape(self.var_weighting, [-1, 1, 1]), axis=0, keepdims=False)
        weighted_stdev = np.sqrt(weighted_vars) * self.scaling
        return uncal_preds, weighted_stdev.tolist()


class PlattCalibrator(UncertaintyCalibrator):
    def __init__(
        self,
        uncertainty_method: str,
        interval_percentile: int,
        regression_calibrator_metric: str,
        calibration_data: MoleculeDataset,
        models: Iterator[MoleculeModel],
        scalers: Iterator[StandardScaler],
        dataset_type: str,
        loss_function: str,
        batch_size: int,
        num_workers: int,
        dropout_sampling_size: int,
    ):
        super().__init__(
            uncertainty_method=uncertainty_method,
            interval_percentile=interval_percentile,
            regression_calibrator_metric=regression_calibrator_metric,
            calibration_data=calibration_data,
            models=models,
            scalers=scalers,
            dataset_type=dataset_type,
            loss_function=loss_function,
            batch_size=batch_size,
            num_workers=num_workers,
            dropout_sampling_size=dropout_sampling_size,
        )
        self.label = f'{uncertainty_method}_platt_confidence'

    def raise_argument_errors(self):
        super().raise_argument_errors()
        if self.dataset_type != 'classification':
            raise ValueError('Platt scaling is only implemented for classification dataset types.')

    def calibrate(self):
        uncal_preds = np.array(self.calibration_predictor.get_uncal_preds()) # shape(data, tasks)
        targets = np.array(self.calibration_data.targets())
        num_tasks = targets.shape[1]
        # If train class sizes are available, set Bayes corrected calibration targets
        if self.calibration_predictor.models[0].train_class_sizes is not None:
            class_size_correction = True
            train_class_sizes = []
            for m in self.calibration_predictor.models:
                train_class_sizes.append(m.train_class_sizes)
            train_class_sizes = np.sum(train_class_sizes, axis = 0) # shape(tasks, 2)
            negative_target = 1/(train_class_sizes[:,0] + 2)
            positive_target = (train_class_sizes[:,1] + 1)/(train_class_sizes[:,1] + 2)
            print('Platt scaling for calibration uses Bayesian correction against training set overfitting, '
                f'replacing calibration targets [0,1] with adjusted values.')
        else:
            class_size_correction = False
            print('Class sizes used in training models unavailable in checkpoints before Chemprop v1.5.0. '
            'No Bayesian correction perfomed as part of class scaling.')

        platt_parameters = []
        for i in range(num_tasks):
            task_targets = targets[:, i]
            task_preds = uncal_preds[:, i]
            if class_size_correction:
                task_targets[task_targets == 0] = negative_target[i]
                task_targets[task_targets == 1] = positive_target[i]
                print(f'Platt Bayesian correction for task {i} in calibration replacing [0,1] targets with {[negative_target[i], positive_target[i]]}')

            def objective(parameters: np.ndarray):
                a = parameters[0]
                b = parameters[1]
                scaled_preds = expit(a * logit(task_preds) + b)
                nll = -1 * np.sum(task_targets * np.log(scaled_preds) + (1 - task_targets) * np.log(1 - scaled_preds))
                return nll

            initial_guess = [1, 0]
            sol = fmin(objective, initial_guess)
            platt_parameters.append(sol)

        platt_parameters = np.array(platt_parameters) # shape(task, 2)
        self.platt_a = platt_parameters[:,0]
        self.platt_b = platt_parameters[:,1]

    def apply_calibration(self, uncal_predictor: UncertaintyPredictor):
        uncal_preds = uncal_predictor.get_uncal_preds() # shape(data, task)
        cal_preds = expit(np.expand_dims(self.platt_a, axis=0) * logit(uncal_preds) + np.expand_dims(self.platt_b, axis=0))
        return uncal_preds, cal_preds


class IsotonicCalibrator(UncertaintyCalibrator):
    def __init__(
        self,
        uncertainty_method: str,
        interval_percentile: int,
        regression_calibrator_metric: str,
        calibration_data: MoleculeDataset,
        models: Iterator[MoleculeModel],
        scalers: Iterator[StandardScaler],
        dataset_type: str,
        loss_function: str,
        batch_size: int,
        num_workers: int,
        dropout_sampling_size: int,
    ):
        super().__init__(
            uncertainty_method=uncertainty_method,
            interval_percentile=interval_percentile,
            regression_calibrator_metric=regression_calibrator_metric,
            calibration_data=calibration_data,
            models=models,
            scalers=scalers,
            dataset_type=dataset_type,
            loss_function=loss_function,
            batch_size=batch_size,
            num_workers=num_workers,
            dropout_sampling_size=dropout_sampling_size,
        )
        self.label = f'{uncertainty_method}_isotonic_confidence'

    def raise_argument_errors(self):
        super().raise_argument_errors()
        if self.dataset_type != 'classification':
            raise ValueError('Isotonic Regression is only implemented for classification dataset types.')

    def calibrate(self):
        uncal_preds = np.array(self.calibration_predictor.get_uncal_preds()) # shape(data, tasks)
        targets = np.array(self.calibration_data.targets())
        num_tasks = targets.shape[1]

        isotonic_models = []
        for i in range(num_tasks):
            task_targets = targets[:, i]
            task_preds = uncal_preds[:, i]

            isotonic_model = IsotonicRegression(y_min=0, y_max=1, out_of_bounds='clip')
            isotonic_model.fit(task_preds,task_targets)
            isotonic_models.append(isotonic_model)

        self.isotonic_models = isotonic_models

    def apply_calibration(self, uncal_predictor: UncertaintyPredictor):
        uncal_preds = uncal_predictor.get_uncal_preds() # shape(data, task)
        transpose_cal_preds = []
        for i, iso_model in enumerate(self.isotonic_models):
            task_preds = uncal_preds[:,i]
            task_cal = iso_model.predict(task_preds)
            transpose_cal_preds.append(task_cal)
        cal_preds = np.transpose(transpose_cal_preds)
        return uncal_preds, cal_preds


class IsotonicMulticlassCalibrator(UncertaintyCalibrator):
    def __init__(
        self,
        uncertainty_method: str,
        interval_percentile: int,
        regression_calibrator_metric: str,
        calibration_data: MoleculeDataset,
        models: Iterator[MoleculeModel],
        scalers: Iterator[StandardScaler],
        dataset_type: str,
        loss_function: str,
        batch_size: int,
        num_workers: int,
        dropout_sampling_size: int,
    ):
        super().__init__(
            uncertainty_method=uncertainty_method,
            interval_percentile=interval_percentile,
            regression_calibrator_metric=regression_calibrator_metric,
            calibration_data=calibration_data,
            models=models,
            scalers=scalers,
            dataset_type=dataset_type,
            loss_function=loss_function,
            batch_size=batch_size,
            num_workers=num_workers,
            dropout_sampling_size=dropout_sampling_size,
        )
        self.label = f'{uncertainty_method}_isotonic_confidence'

    def raise_argument_errors(self):
        super().raise_argument_errors()
        if self.dataset_type != 'multiclass':
            raise ValueError('Isotonic Multiclass Regression is only implemented for multiclass dataset types.')

    def calibrate(self):
        uncal_preds = np.array(self.calibration_predictor.get_uncal_preds()) # shape(data, tasks, num_classes)
        targets = np.array(self.calibration_data.targets()) # shape(data, tasks)
        self.num_tasks = targets.shape[1]
        self.num_classes = uncal_preds.shape[2]

        isotonic_models = []
        for i in range(self.num_tasks):
            isotonic_models.append([])
            task_targets = targets[:, i] # shape(data)
            for j in range(self.num_classes):
                class_preds = uncal_preds[:, i, j] # shape(data)
                positive_class_targets = task_targets == j
                
                class_targets = np.ones_like(class_preds)
                class_targets[positive_class_targets] = 1
                class_targets[~positive_class_targets] = 0

                isotonic_model = IsotonicRegression(y_min=0, y_max=1, out_of_bounds='clip')
                isotonic_model.fit(class_preds,class_targets)
                isotonic_models[i].append(isotonic_model)

        self.isotonic_models = isotonic_models # shape(tasks, classes)

    def apply_calibration(self, uncal_predictor: UncertaintyPredictor):
        uncal_preds = uncal_predictor.get_uncal_preds() # shape(data, task, class)
        transpose_cal_preds = []
        for i in range(self.num_tasks):
            transpose_cal_preds.append([])
            for j in range(self.num_classes):
                class_preds = uncal_preds[:,i,j]
                class_cal = self.isotonic_models[i][j].predict(class_preds)
                transpose_cal_preds[i].append(class_cal) # shape (task, class, data)
        cal_preds = np.transpose(transpose_cal_preds, [2, 0, 1]) # shape(data, task, class)
        cal_preds = cal_preds / np.sum(cal_preds, axis=2, keepdims=True)
        return uncal_preds, cal_preds


def uncertainty_calibrator_builder(
    calibration_method: str,
    uncertainty_method: str,
    regression_calibrator_metric: str,
    interval_percentile: int,
    calibration_data: MoleculeDataset,
    models: Iterator[MoleculeModel],
    scalers: Iterator[StandardScaler],
    dataset_type: str,
    loss_function: str,
    batch_size: int,
    num_workers: int,
    dropout_sampling_size: int,
    ) -> UncertaintyCalibrator:
    """
    
    """
    if calibration_method is None:
        if dataset_type == 'regression':
            if regression_calibrator_metric == 'stdev':
                calibration_method = 'zscaling'
            else:
                calibration_method = 'zelikman_interval'
        if dataset_type in ['classification', 'multiclass']:
            calibration_method == 'isotonic'


    supported_calibrators = {
        'zscaling': ZScalingCalibrator,
        'tscaling': TScalingCalibrator,
        'zelikman_interval': ZelikmanCalibrator,
        'mve_weighting': MVEWeightingCalibrator,
        'platt': PlattCalibrator,
        'isotonic': IsotonicCalibrator if dataset_type == 'classification' else IsotonicMulticlassCalibrator,
    }

    calibrator_class = supported_calibrators.get(calibration_method, None)
    
    if calibrator_class is None:
        raise NotImplementedError(f'Calibrator type {calibration_method} is not currently supported. Avalable options are: {list(supported_calibrators.keys())}')
    else:
        calibrator = calibrator_class(
            uncertainty_method=uncertainty_method,
            regression_calibrator_metric=regression_calibrator_metric,
            interval_percentile=interval_percentile,
            calibration_data=calibration_data,
            models=models,
            scalers=scalers,
            dataset_type=dataset_type,
            loss_function=loss_function,
            batch_size=batch_size,
            num_workers=num_workers,
            dropout_sampling_size=dropout_sampling_size,
        )
    return calibrator