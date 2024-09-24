from abc import ABC, abstractmethod

from torch import Tensor

from chemprop.utils.registry import ClassRegistry


class EvaluatorBase(ABC):
    """A base class to evaluate the quality of uncertainty estimates using a specified metric."""

    @abstractmethod
    def evalute(self, *args, **kwargs) -> Tensor:
        """
        Evaluate the performance of uncertainty predictions against the model target values.
        """
        pass


UncertaintyEvaluatorRegistry = ClassRegistry[EvaluatorBase]()


class RegressionEvaluator(EvaluatorBase):
    """Evaluates the quality of uncertainty estimates in regression tasks."""

    @abstractmethod
    def evaluate(self, preds: Tensor, uncs: Tensor, targets: Tensor, mask: Tensor) -> Tensor:
        """Evaluate the performance of uncertainty predictions against the model target values.

        Parameters
        ----------
        preds: Tensor
            the predictions for regression tasks. It is a tensor of the shape of ``n x t``, where ``n`` is the number of input
            molecules/reactions, and ``t`` is the number of tasks.
        uncs: Tensor
            the predicted uncertainties (variance) of the shape of ``n x t``
        targets: Tensor
            a tensor of the shape ``n x t``
        mask: Tensor
            a tensor of the shape ``n x t`` indicating whether the given values should be used in the fitting

        Returns
        -------
        Tensor
            a tensor of the shape ``t`` containing the evaluated metrics
        """


@UncertaintyEvaluatorRegistry.register("nll-regression")
class NLLRegressionEvaluator(RegressionEvaluator):
    def evaluate(self, preds: Tensor, uncs: Tensor, targets: Tensor, mask: Tensor) -> Tensor:
        ...
        return


@UncertaintyEvaluatorRegistry.register("miscalibration_area")
class CalibrationAreaEvaluator(RegressionEvaluator):
    def evaluate(self, preds: Tensor, uncs: Tensor, targets: Tensor, mask: Tensor) -> Tensor:
        ...
        return


@UncertaintyEvaluatorRegistry.register("ence")
class ExpectedNormalizedErrorEvaluator(RegressionEvaluator):
    def evaluate(self, preds: Tensor, uncs: Tensor, targets: Tensor, mask: Tensor) -> Tensor:
        ...
        return


@UncertaintyEvaluatorRegistry.register("spearman")
class SpearmanEvaluator(RegressionEvaluator):
    def evaluate(self, preds: Tensor, uncs: Tensor, targets: Tensor, mask: Tensor) -> Tensor:
        ...
        return


@UncertaintyEvaluatorRegistry.register("conformal-coverage-regression")
class ConformalRegressionEvaluator(RegressionEvaluator):
    def evaluate(self, preds: Tensor, uncs: Tensor, targets: Tensor, mask: Tensor) -> Tensor:
        ...
        return


class BinaryClassificationEvaluator(EvaluatorBase):
    """Evaluates the quality of uncertainty estimates in binary classification tasks."""

    @abstractmethod
    def evaluate(self, uncs: Tensor, targets: Tensor, mask: Tensor) -> Tensor:
        """Evaluate the performance of uncertainty predictions against the model target values.

        Parameters
        ----------
        uncs: Tensor
            the predicted uncertainties (i.e., the predicted probability of class 1) of the shape of ``n x t``, where ``n`` is the number of input
            molecules/reactions, and ``t`` is the number of tasks.
        targets: Tensor
            a tensor of the shape ``n x t``
        mask: Tensor
            a tensor of the shape ``n x t`` indicating whether the given values should be used in the fitting

        Returns
        -------
        Tensor
            a tensor of the shape ``t`` containing the evaluated metrics
        """


@UncertaintyEvaluatorRegistry.register("nll-classification")
class NLLClassEvaluator(BinaryClassificationEvaluator):
    def evaluate(self, uncs: Tensor, targets: Tensor, mask: Tensor) -> Tensor:
        ...
        return


@UncertaintyEvaluatorRegistry.register("conformal-coverage-classification")
class ConformalMultilabelEvaluator(BinaryClassificationEvaluator):
    def evaluate(self, uncs: Tensor, targets: Tensor, mask: Tensor) -> Tensor:
        ...
        return


class MulticlassClassificationEvaluator(EvaluatorBase):
    """Evaluates the quality of uncertainty estimates in multiclass classification tasks."""

    @abstractmethod
    def evaluate(self, uncs: Tensor, targets: Tensor, mask: Tensor) -> Tensor:
        """Evaluate the performance of uncertainty predictions against the model target values.

        Parameters
        ----------
        uncs: Tensor
            the predicted uncertainties (i.e., the predicted probabilities for each class) of the shape of ``n x t x c``, where ``n`` is the number of input
            molecules/reactions, ``t`` is the number of tasks, and ``c`` is the number of classes.
        targets: Tensor
            a tensor of the shape ``n x t``
        mask: Tensor
            a tensor of the shape ``n x t`` indicating whether the given values should be used in the fitting

        Returns
        -------
        Tensor
            a tensor of the shape ``t`` containing the evaluated metrics
        """


@UncertaintyEvaluatorRegistry.register("nll-multiclass")
class NLLMultiEvaluator(MulticlassClassificationEvaluator):
    def evaluate(self, uncs: Tensor, targets: Tensor, mask: Tensor) -> Tensor:
        ...
        return


@UncertaintyEvaluatorRegistry.register("conformal-coverage-multiclass")
class ConformalMulticlassEvaluator(MulticlassClassificationEvaluator):
    def evaluate(self, uncs: Tensor, targets: Tensor, mask: Tensor) -> Tensor:
        ...
        return
