from typing import Any, Type

from chemprop.nn import loss, predictors

__all__ = ["pop_attr"]


def pop_attr(o: object, attr: str, *args) -> Any | None:
    """like ``pop()`` but for attribute maps"""
    match len(args):
        case 0:
            return _pop_attr(o, attr)
        case 1:
            return _pop_attr_d(o, attr, args[0])
        case _:
            raise TypeError(f"Expected at most 2 arguments! got: {len(args)}")


def _pop_attr(o: object, attr: str) -> Any:
    val = getattr(o, attr)
    delattr(o, attr)

    return val


def _pop_attr_d(o: object, attr: str, default: Any | None = None) -> Any | None:
    try:
        val = getattr(o, attr)
        delattr(o, attr)
    except AttributeError:
        val = default

    return val


def validate_loss_function(
    predictor_ffn: Type[predictors._FFNPredictorBase], criterion: Type[loss.LossFunction]
):
    match predictor_ffn:
        case predictors.RegressionFFN:
            if criterion not in (loss.MSELoss, loss.BoundedMSELoss):
                raise ValueError(f"Expected a regression loss function! got: {criterion.__name__}")
        case predictors.MveFFN:
            if criterion is not loss.MVELoss:
                raise ValueError(f"Expected a MVE loss function! got: {criterion.__name__}")
        case predictors.EvidentialFFN:
            if criterion is not loss.EvidentialLoss:
                raise ValueError(f"Expected an evidential loss function! got: {criterion.__name__}")
        case predictors.BinaryClassificationFFN:
            if criterion not in (loss.BCELoss, loss.BinaryMCCLoss):
                raise ValueError(
                    f"Expected a binary classification loss function! got: {criterion.__name__}"
                )
        case predictors.BinaryDirichletFFN:
            if loss is not loss.BinaryDirichletLoss:
                raise ValueError(
                    f"Expected a binary Dirichlet loss function! got: {criterion.__name__}"
                )
        case predictors.MulticlassClassificationFFN:
            if loss not in (loss.CrossEntropyLoss, loss.MulticlassMCCLoss):
                raise ValueError(
                    f"Expected a multiclass classification loss function! got: {criterion.__name__}"
                )
        case predictors.MulticlassDirichletFFN:
            if loss is not loss.MulticlassDirichletLoss:
                raise ValueError(
                    f"Expected a multiclass Dirichlet loss function! got: {criterion.__name__}"
                )
        case predictors.SpectralFFN:
            if loss not in (loss.SIDLoss, loss.WassersteinLoss):
                raise ValueError(f"Expected a spectral loss function! got: {criterion.__name__}")
        case _:
            raise ValueError(
                f"Unknown predictor function! got: {predictor_ffn}. "
                f"Expected one of: {tuple(predictors.PredictorRegistry.values())}"
            )
