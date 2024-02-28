from abc import ABC, abstractmethod
from typing import List

import numpy as np
from scipy.stats import t, spearmanr
from scipy.special import erfinv

from chemprop.uncertainty.uncertainty_calibrator import UncertaintyCalibrator
from chemprop.train import evaluate_predictions


class UncertaintyEvaluator(ABC):
    """
    A class for evaluating the effectiveness of uncertainty estimates with metrics.
    """

    def __init__(
        self,
        evaluation_method: str,
        calibration_method: str,
        uncertainty_method: str,
        dataset_type: str,
        loss_function: str,
        calibrator: UncertaintyCalibrator,
        is_atom_bond_targets: bool,
    ):
        self.evaluation_method = evaluation_method
        self.calibration_method = calibration_method
        self.uncertainty_method = uncertainty_method
        self.dataset_type = dataset_type
        self.loss_function = loss_function
        self.calibrator = calibrator
        self.is_atom_bond_targets = is_atom_bond_targets

        self.raise_argument_errors()

    def raise_argument_errors(self):
        """
        Raise errors for incompatibilities between dataset type and uncertainty method, or similar.
        """
        if self.dataset_type == "spectra":
            raise ValueError(
                "No uncertainty evaluators implemented for spectra dataset type."
            )
        if self.uncertainty_method in ["ensemble", "dropout"] and self.dataset_type in [
            "classification",
            "multiclass",
        ]:
            raise ValueError(
                "Though ensemble and dropout uncertainty methods are available for classification \
                    multiclass dataset types, their outputs are not confidences and are not \
                    compatible with any implemented evaluation methods for classification."
            )
        if self.uncertainty_method == "dirichlet":
            raise ValueError(
                "The Dirichlet uncertainty method returns an evidential uncertainty value rather than a \
                    class confidence. It is not compatible with any implemented evaluation methods. \
                    To evaluate the performance of a model trained using the Dirichlet loss function, \
                    use the classification uncertainty method in a separate job."
            )

    @abstractmethod
    def evaluate(
        self,
        targets: List[List[float]],
        preds: List[List[float]],
        uncertainties: List[List[float]],
        mask: List[List[bool]],
    ) -> List[float]:
        """
        Evaluate the performance of uncertainty predictions against the model target values.

        :param targets:  The target values for prediction.
        :param preds: The prediction values of a model on the test set.
        :param uncertainties: The estimated uncertainty values, either calibrated or uncalibrated, of a model on the test set.
        :param mask: Whether the values in targets were provided.

        :return: A list of metric values for each model task.
        """


class MetricEvaluator(UncertaintyEvaluator):
    """
    A class for evaluating confidence estimates of classification and multiclass datasets using builtin evaluation metrics.
    """

    def evaluate(
        self,
        targets: List[List[float]],
        preds: List[List[float]],
        uncertainties: List[List[float]],
        mask: List[List[bool]],
    ):
        return evaluate_predictions(
            preds=uncertainties,
            targets=targets,
            num_tasks=np.array(targets).shape[1],
            metrics=[self.evaluation_method],
            dataset_type=self.dataset_type,
        )[self.evaluation_method]


class NLLRegressionEvaluator(UncertaintyEvaluator):
    """
    A class for evaluating regression uncertainty values using the mean negative-log-likelihood
    of the actual targets given the probability distributions estimated by the model.
    """

    def raise_argument_errors(self):
        super().raise_argument_errors()
        if self.dataset_type != "regression":
            raise ValueError(
                "NLL Regression Evaluator is only for regression dataset types."
            )

    def evaluate(
        self,
        targets: List[List[float]],
        preds: List[List[float]],
        uncertainties: List[List[float]],
        mask: List[List[bool]],
    ):
        if self.calibrator is None:  # uncalibrated regression uncertainties are variances
            uncertainties = np.array(uncertainties)
            preds = np.array(preds)
            targets = np.array(targets)
            mask = np.array(mask)
            num_tasks = len(mask)
            if self.is_atom_bond_targets:
                uncertainties = [np.concatenate(x) for x in zip(*uncertainties)]
                preds = [np.concatenate(x) for x in zip(*preds)]
                targets = [np.concatenate(x) for x in zip(*targets)]
            else:
                uncertainties = np.array(list(zip(*uncertainties)))
                preds = np.array(list(zip(*preds)))
                targets = targets.astype(float)
                targets = np.array(list(zip(*targets)))
            nll = []
            for i in range(num_tasks):
                task_mask = mask[i]
                task_unc = uncertainties[i][task_mask]
                task_preds = preds[i][task_mask]
                task_targets = targets[i][task_mask]
                task_nll = np.log(2 * np.pi * task_unc) / 2 \
                    + (task_preds - task_targets) ** 2 / (2 * task_unc)
                nll.append(task_nll.mean())
            return nll
        else:
            nll = self.calibrator.nll(
                preds=preds, unc=uncertainties, targets=targets, mask=mask
            )  # shape(data, task)
            return nll


class NLLClassEvaluator(UncertaintyEvaluator):
    """
    A class for evaluating classification uncertainty values using the mean negative-log-likelihood
    of the actual targets given the probabilities assigned to them by the model.
    """

    def raise_argument_errors(self):
        super().raise_argument_errors()
        if self.dataset_type != "classification":
            raise ValueError(
                "NLL Classification Evaluator is only for classification dataset types."
            )

    def evaluate(
        self,
        targets: List[List[float]],
        preds: List[List[float]],
        uncertainties: List[List[float]],
        mask: List[List[bool]],
    ):
        targets = np.array(targets)
        mask = np.array(mask)
        num_tasks = len(mask)
        uncertainties = np.array(uncertainties)
        if self.is_atom_bond_targets:
            uncertainties = [np.concatenate(x) for x in zip(*uncertainties)]
            targets = [np.concatenate(x) for x in zip(*targets)]
        else:
            uncertainties = np.array(list(zip(*uncertainties)))
            targets = targets.astype(float)
            targets = np.array(list(zip(*targets)))
        nll = []
        for i in range(num_tasks):
            task_mask = mask[i]
            task_unc = uncertainties[i][task_mask]
            task_targets = targets[i][task_mask]
            task_likelihood = task_unc * task_targets + (1 - task_unc) * (1 - task_targets)
            task_nll = -1 * np.log(task_likelihood)
            nll.append(task_nll.mean())
        return nll


class NLLMultiEvaluator(UncertaintyEvaluator):
    """
    A class for evaluating multiclass uncertainty values using the mean negative-log-likelihood
    of the actual targets given the probabilities assigned to them by the model.
    """

    def raise_argument_errors(self):
        super().raise_argument_errors()
        if self.dataset_type != "multiclass":
            raise ValueError(
                "NLL Multiclass Evaluator is only for multiclass dataset types."
            )

    def evaluate(
        self,
        targets: List[List[float]],
        preds: List[List[float]],
        uncertainties: List[List[float]],
        mask: List[List[bool]],
    ):
        targets = np.array(targets, dtype=int)  # shape(data, tasks)
        mask = np.array(mask)
        num_tasks = len(mask)
        uncertainties = np.array(uncertainties)
        nll = []
        for i in range(num_tasks):
            task_mask = mask[:, i]
            task_preds = uncertainties[task_mask, i]
            task_targets = targets[task_mask, i]  # shape(data)
            bin_targets = np.zeros_like(task_preds)  # shape(data, classes)
            bin_targets[np.arange(task_targets.shape[0]), task_targets] = 1
            task_likelihood = np.sum(bin_targets * task_preds, axis=1)
            task_nll = -1 * np.log(task_likelihood)
            nll.append(task_nll.mean())
        return nll


class CalibrationAreaEvaluator(UncertaintyEvaluator):
    """
    A class for evaluating regression uncertainty values based on how they deviate from perfect
    calibration on an observed-probability versus expected-probability plot.
    """

    def raise_argument_errors(self):
        super().raise_argument_errors()
        if self.dataset_type != "regression":
            raise ValueError(
                "Miscalibration area is only implemented for regression dataset types."
            )

    def evaluate(
        self,
        targets: List[List[float]],
        preds: List[List[float]],
        uncertainties: List[List[float]],
        mask: List[List[bool]],
    ):
        targets = np.array(targets)  # shape(data, tasks)
        mask = np.array(mask)
        num_tasks = len(mask)
        uncertainties = np.array(uncertainties)
        preds = np.array(preds)

        if self.is_atom_bond_targets:
            uncertainties = [np.concatenate(x) for x in zip(*uncertainties)]
            targets = [np.concatenate(x) for x in zip(*targets)]
            preds = [np.concatenate(x) for x in zip(*preds)]
        else:
            uncertainties = np.array(list(zip(*uncertainties)))
            targets = targets.astype(float)
            targets = np.array(list(zip(*targets)))
            preds = np.array(list(zip(*preds)))
        # using 101 bin edges, hardcoded
        fractions = np.zeros([num_tasks, 101])  # shape(tasks, 101)
        fractions[:, 100] = 1

        if self.calibrator is not None:
            original_metric = self.calibrator.regression_calibrator_metric
            original_scaling = self.calibrator.scaling
            original_interval = self.calibrator.interval_percentile

            bin_scaling = [0]

            for i in range(1, 100):
                self.calibrator.regression_calibrator_metric = "interval"
                self.calibrator.interval_percentile = i
                self.calibrator.calibrate()
                bin_scaling.append(self.calibrator.scaling)

            for j in range(num_tasks):
                task_mask = mask[j]
                task_targets = targets[j][task_mask]
                task_preds = preds[j][task_mask]
                task_error = np.abs(task_preds - task_targets)
                task_unc = uncertainties[j][task_mask]

                for i in range(1, 100):
                    bin_unc = task_unc / original_scaling[j] * bin_scaling[i][j]
                    bin_fraction = np.mean(bin_unc >= task_error)
                    fractions[j, i] = bin_fraction

            # return calibration settings to original state
            self.calibrator.regression_calibrator_metric = original_metric
            self.calibrator.scaling = original_scaling
            self.calibrator.interval_percentile = original_interval

        else:  # uncertainties are uncalibrated variances
            bin_scaling = [0]
            for i in range(1, 100):
                bin_scaling.append(erfinv(i / 100) * np.sqrt(2))
            for j in range(num_tasks):
                task_mask = mask[j]
                task_targets = targets[j][task_mask]
                task_preds = preds[j][task_mask]
                task_error = np.abs(task_preds - task_targets)
                task_unc = uncertainties[j][task_mask]
                for i in range(1, 100):
                    bin_unc = np.sqrt(task_unc) * bin_scaling[i]
                    bin_fraction = np.mean(bin_unc >= task_error)
                    fractions[j, i] = bin_fraction

        # trapezoid rule
        auce = np.sum(
            0.01 * np.abs(fractions - np.expand_dims(np.arange(101) / 100, axis=0)),
            axis=1,
        )
        return auce.tolist()


class ExpectedNormalizedErrorEvaluator(UncertaintyEvaluator):
    """
    A class that evaluates uncertainty performance by binning together clusters of predictions
    and comparing the average predicted variance of the clusters against the RMSE of the cluster.
    Method discussed in https://doi.org/10.1021/acs.jcim.9b00975.
    """

    def raise_argument_errors(self):
        super().raise_argument_errors()
        if self.dataset_type != "regression":
            raise ValueError(
                "Expected normalized error is only appropriate for regression dataset types."
            )

    def evaluate(
        self,
        targets: List[List[float]],
        preds: List[List[float]],
        uncertainties: List[List[float]],
        mask: List[List[bool]],
    ):
        targets = np.array(targets)  # shape(data, tasks)
        mask = np.array(mask)
        num_tasks = len(mask)
        uncertainties = np.array(uncertainties)
        preds = np.array(preds)
        if self.is_atom_bond_targets:
            uncertainties = [np.concatenate(x) for x in zip(*uncertainties)]
            targets = [np.concatenate(x) for x in zip(*targets)]
            preds = [np.concatenate(x) for x in zip(*preds)]
        else:
            uncertainties = np.array(list(zip(*uncertainties)))
            targets = targets.astype(float)
            targets = np.array(list(zip(*targets)))
            preds = np.array(list(zip(*preds)))
        # get stdev scaling then revert if interval
        if self.calibrator is not None:
            original_metric = self.calibrator.regression_calibrator_metric
            original_scaling = self.calibrator.scaling
            if (
                self.calibration_method != "tscaling"
                and self.calibrator.regression_calibrator_metric == "interval"
            ):
                self.calibrator.regression_calibrator_metric = "stdev"
                self.calibrator.calibrate()
                stdev_scaling = self.calibrator.scaling
                self.calibrator.regression_calibrator_metric = original_metric
                self.calibrator.scaling = original_scaling

        root_mean_vars = np.zeros([num_tasks, 100])  # shape(tasks, 100)
        rmses = np.zeros_like(root_mean_vars)

        for i in range(num_tasks):
            task_mask = mask[i]  # shape(data)
            task_targets = targets[i][task_mask]
            task_preds = preds[i][task_mask]
            task_error = np.abs(task_preds - task_targets)
            task_unc = uncertainties[i][task_mask]

            sort_idx = np.argsort(task_unc)
            task_unc = task_unc[sort_idx]
            task_error = task_error[sort_idx]

            # 100 bins
            split_unc = np.array_split(task_unc, 100)  # shape(list100, data/100)
            split_error = np.array_split(task_error, 100)

            for j in range(100):
                if self.calibrator is None:  # starts as a variance
                    root_mean_vars[i, j] = np.sqrt(np.mean(split_unc[j]))
                    rmses[i, j] = np.sqrt(np.mean(np.square(split_error[j])))
                elif self.calibration_method == "tscaling":  # convert back to sample stdev
                    bin_unc = split_unc[j] / original_scaling[i]
                    bin_var = t.var(df=self.calibrator.num_models - 1, scale=bin_unc)
                    root_mean_vars[i, j] = np.sqrt(np.mean(bin_var))
                    rmses[i, j] = np.sqrt(np.mean(np.square(split_error[j])))
                else:
                    bin_unc = split_unc[j]
                    if self.calibrator.regression_calibrator_metric == "interval":
                        bin_unc = bin_unc / original_scaling[i] * stdev_scaling[i]  # convert from interval to stdev as needed
                    root_mean_vars[i, j] = np.sqrt(np.mean(np.square(bin_unc)))
                    rmses[i, j] = np.sqrt(np.mean(np.square(split_error[j])))

        ence = np.mean(np.abs(root_mean_vars - rmses) / root_mean_vars, axis=1)
        return ence.tolist()


class SpearmanEvaluator(UncertaintyEvaluator):
    """
    Class evaluating uncertainty performance using the spearman rank correlation. Method produces
    better scores (closer to 1 in the [-1, 1] range) when the uncertainty values are predictive
    of the ranking of prediciton errors.
    """

    def raise_argument_errors(self):
        super().raise_argument_errors()
        if self.dataset_type != "regression":
            raise ValueError(
                "Spearman rank correlation is only appropriate for regression dataset types."
            )

    def evaluate(
        self,
        targets: List[List[float]],
        preds: List[List[float]],
        uncertainties: List[List[float]],
        mask: List[List[bool]],
    ):
        targets = np.array(targets)  # shape(data, tasks)
        uncertainties = np.array(uncertainties)
        mask = np.array(mask)
        num_tasks = len(mask)
        preds = np.array(preds)
        spearman_coeffs = []
        if self.is_atom_bond_targets:
            uncertainties = [np.concatenate(x) for x in zip(*uncertainties)]
            targets = [np.concatenate(x) for x in zip(*targets)]
            preds = [np.concatenate(x) for x in zip(*preds)]
        else:
            uncertainties = np.array(list(zip(*uncertainties)))
            targets = targets.astype(float)
            targets = np.array(list(zip(*targets)))
            preds = np.array(list(zip(*preds)))
        for i in range(num_tasks):
            task_mask = mask[i]
            task_unc = uncertainties[i][task_mask]
            task_targets = targets[i][task_mask]
            task_preds = preds[i][task_mask]
            task_error = np.abs(task_preds - task_targets)
            spmn = spearmanr(task_unc, task_error).correlation
            spearman_coeffs.append(spmn)
        return spearman_coeffs


class ConformalRegressionEvaluator(UncertaintyEvaluator):
    """
    A class for evaluating the coverage of conformal regression intervals.
    """

    def raise_argument_errors(self):
        super().raise_argument_errors()
        if self.dataset_type != "regression":
            raise ValueError(
                "Conformal Regression Evaluator is only for regression dataset types."
            )

    def evaluate(
        self,
        targets: List[List[float]],
        preds: List[List[float]],
        uncertainties: List[List[float]],
        mask: List[List[bool]],
    ):
        """
        Args:
            targets: shape(data, tasks)
            preds: shape(data, tasks)
            uncertainties: shape(data, tasks)
            mask: shape(data, tasks)

        Returns:
            Conformal coverage for each task
        """
        uncertainties = np.array(uncertainties)
        targets = np.array(targets)
        preds = np.array(preds)
        mask = np.array(mask)
        num_tasks = uncertainties.shape[1]
        if self.is_atom_bond_targets:
            uncertainties = [np.concatenate(x) for x in zip(*uncertainties)]
            targets = [np.concatenate(x) for x in zip(*targets)]
            preds = [np.concatenate(x) for x in zip(*preds)]
        else:
            uncertainties = np.array(list(zip(*uncertainties)))
            targets = targets.astype(float)
            targets = np.array(list(zip(*targets)))
            preds = np.array(list(zip(*preds)))

        results = []
        for i in range(num_tasks):
            task_mask = mask[i]
            task_unc = uncertainties[i][task_mask]
            task_targets = targets[i][task_mask]
            task_preds = preds[i][task_mask]
            unc_task_lower = task_preds - task_unc
            unc_task_upper = task_preds + task_unc
            task_results = np.logical_and(unc_task_lower <= task_targets, task_targets <= unc_task_upper)
            results.append(task_results.sum() / task_results.shape[0])

        return results


class ConformalMulticlassEvaluator(UncertaintyEvaluator):
    """
    A class for evaluating the coverage of conformal prediction on multiclass datasets.
    """

    def raise_argument_errors(self):
        super().raise_argument_errors()
        if self.dataset_type != "multiclass":
            raise ValueError(
                "Conformal Multiclass Evaluator is only for multiclass dataset types."
            )

    def evaluate(
        self,
        targets: List[List[float]],
        preds: List[List[float]],
        uncertainties: List[List[float]],
        mask: List[List[bool]],
    ):
        """
        Args:
            targets: shape(data, tasks)
            preds: shape(data, tasks, num_classes)
            uncertainties: shape(data, tasks, num_classes)
            mask: shape(data, tasks)

        Returns:
            Conformal coverage for each task
        """
        targets = np.array(targets, dtype=float)
        mask = np.array(mask, dtype=bool)
        uncertainties = np.array(uncertainties)
        num_tasks = targets.shape[1]
        results = []

        for i in range(num_tasks):
            task_mask = mask[i]
            task_results = np.take_along_axis(
                uncertainties[task_mask, i], targets[task_mask, i].reshape(-1, 1).astype(int), axis=1
            ).squeeze(1)
            results.append(task_results.sum() / len(task_results))

        return results


class ConformalMultilabelEvaluator(UncertaintyEvaluator):
    """
    A class for evaluating the coverage of conformal prediction on multilabel datasets.
    """

    def raise_argument_errors(self):
        super().raise_argument_errors()
        if self.dataset_type != "classification":
            raise ValueError(
                "Conformal Multilabel Evaluator is only for classification dataset types."
            )

    def evaluate(
        self,
        targets: List[List[float]],
        preds: List[List[float]],
        uncertainties: List[List[float]],
        mask: List[List[bool]],
    ):
        """
        Args:
            targets: shape(data, tasks)
            preds: shape(data, tasks)
            uncertainties: shape(data, tasks)
            mask: shape(data, tasks)

        Returns:
            Conformal coverage for each task
        """
        targets = np.array(targets, dtype=float)
        uncertainties = np.array(uncertainties)
        mask = np.array(mask)
        num_tasks = len(mask)
        if self.is_atom_bond_targets:
            uncertainties = [np.concatenate(x) for x in zip(*uncertainties)]
            targets = [np.concatenate(x) for x in zip(*targets)]
        else:
            uncertainties = np.array(list(zip(*uncertainties)))
            targets = targets.astype(float)
            targets = np.array(list(zip(*targets)))
        results = []
        for i in range(num_tasks):
            task_mask = mask[i]
            task_targets = targets[i][task_mask]
            task_unc_in = uncertainties[i][task_mask]
            task_unc_out = uncertainties[i + num_tasks][task_mask]
            task_results = np.logical_and(task_unc_in <= task_targets, task_targets <= task_unc_out)
            results.append(task_results.sum() / task_results.shape[0])

        return results


def build_uncertainty_evaluator(
    evaluation_method: str,
    calibration_method: str,
    uncertainty_method: str,
    dataset_type: str,
    loss_function: str,
    calibrator: UncertaintyCalibrator,
    is_atom_bond_targets: bool,
) -> UncertaintyEvaluator:
    """
    Function that chooses and returns the appropriate :class: `UncertaintyEvaluator` subclass
    for the provided arguments.
    """
    supported_evaluators = {
        "nll": {
            "regression": NLLRegressionEvaluator,
            "classification": NLLClassEvaluator,
            "multiclass": NLLMultiEvaluator,
            "spectra": None,
        }[dataset_type],
        "miscalibration_area": CalibrationAreaEvaluator,
        "ence": ExpectedNormalizedErrorEvaluator,
        "spearman": SpearmanEvaluator,
        "conformal_coverage": {
            "regression": ConformalRegressionEvaluator,
            "multiclass": ConformalMulticlassEvaluator,
            "classification": ConformalMultilabelEvaluator,
        }[dataset_type],
    }

    classification_metrics = [
        "auc",
        "prc-auc",
        "accuracy",
        "binary_cross_entropy",
        "f1",
        "mcc",
    ]
    multiclass_metrics = ["cross_entropy", "accuracy", "f1", "mcc"]
    if dataset_type == "classification" and evaluation_method in classification_metrics:
        evaluator_class = MetricEvaluator
    elif dataset_type == "multiclass" and evaluation_method in multiclass_metrics:
        evaluator_class = MetricEvaluator
    else:
        evaluator_class = supported_evaluators.get(evaluation_method, None)

    if evaluator_class is None:
        raise NotImplementedError(
            f"Evaluator type {evaluation_method} is not supported. Available options are all calibration/multiclass metrics and {list(supported_evaluators.keys())}"
        )
    else:
        evaluator = evaluator_class(
            evaluation_method=evaluation_method,
            calibration_method=calibration_method,
            uncertainty_method=uncertainty_method,
            dataset_type=dataset_type,
            loss_function=loss_function,
            calibrator=calibrator,
            is_atom_bond_targets=is_atom_bond_targets,
        )
        return evaluator
