from typing import Any, Type

from chemprop.nn import metrics, predictors

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
    predictor_ffn: Type[predictors._FFNPredictorBase], criterion: Type[metrics.ChempropMetric]
):
    match predictor_ffn:
        case predictors.RegressionFFN:
            if criterion not in (metrics.MSE):
                raise ValueError(
                    f"Expected a regression metrics function! got: {criterion.__name__}"
                )
        case predictors.MveFFN:
            if criterion is not metrics.MVELoss:
                raise ValueError(f"Expected a MVE metrics function! got: {criterion.__name__}")
        case predictors.EvidentialFFN:
            if criterion is not metrics.EvidentialLoss:
                raise ValueError(
                    f"Expected an evidential metrics function! got: {criterion.__name__}"
                )
        case predictors.BinaryClassificationFFN:
            if criterion not in (metrics.BCELoss, metrics.BinaryMCCLoss):
                raise ValueError(
                    f"Expected a binary classification metrics function! got: {criterion.__name__}"
                )
        case predictors.BinaryDirichletFFN:
            if metrics is not metrics.BinaryDirichletLoss:
                raise ValueError(
                    f"Expected a binary Dirichlet metrics function! got: {criterion.__name__}"
                )
        case predictors.MulticlassClassificationFFN:
            if metrics not in (metrics.CrossEntropyLoss, metrics.MulticlassMCCLoss):
                raise ValueError(
                    f"Expected a multiclass classification metrics function! got: {criterion.__name__}"
                )
        case predictors.MulticlassDirichletFFN:
            if metrics is not metrics.MulticlassDirichletLoss:
                raise ValueError(
                    f"Expected a multiclass Dirichlet metrics function! got: {criterion.__name__}"
                )
        case predictors.SpectralFFN:
            if metrics not in (metrics.SID, metrics.Wasserstein):
                raise ValueError(f"Expected a spectral metrics function! got: {criterion.__name__}")
        case _:
            raise ValueError(
                f"Unknown predictor function! got: {predictor_ffn}. "
                f"Expected one of: {tuple(predictors.PredictorRegistry.values())}"
            )
