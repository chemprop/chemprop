from typing import Any, Type

from chemprop.v2.nn import loss
from chemprop.v2.models.modules import readout

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
    readout_ffn: Type[readout.ReadoutFFNBase], criterion: Type[loss.LossFunction]
):
    match readout_ffn:
        case readout.RegressionFFN:
            if criterion not in (loss.MSELoss, loss.BoundedMSELoss):
                raise ValueError(f"Expected a regression loss function! got: {criterion.__name__}")
        case readout.MveFFN:
            if criterion is not loss.MVELoss:
                raise ValueError(f"Expected a MVE loss function! got: {criterion.__name__}")
        case readout.EvidentialFFN:
            if criterion is not loss.EvidentialLoss:
                raise ValueError(f"Expected an evidential loss function! got: {criterion.__name__}")
        case readout.BinaryClassificationFFN:
            if criterion not in (loss.BCELoss, loss.BinaryMCCLoss):
                raise ValueError(
                    f"Expected a binary classification loss function! got: {criterion.__name__}"
                )
        case readout.BinaryDirichletFFN:
            if loss is not loss.BinaryDirichletLoss:
                raise ValueError(
                    f"Expected a binary Dirichlet loss function! got: {criterion.__name__}"
                )
        case readout.MulticlassClassificationFFN:
            if loss not in (loss.CrossEntropyLoss, loss.MulticlassMCCLoss):
                raise ValueError(
                    f"Expected a multiclass classification loss function! got: {criterion.__name__}"
                )
        case readout.MulticlassDirichletFFN:
            if loss is not loss.MulticlassDirichletLoss:
                raise ValueError(
                    f"Expected a multiclass Dirichlet loss function! got: {criterion.__name__}"
                )
        case readout.SpectralFFN:
            if loss not in (loss.SIDLoss, loss.WassersteinLoss):
                raise ValueError(f"Expected a spectral loss function! got: {criterion.__name__}")
        case _:
            raise ValueError(
                f"Unknown readout function! got: {readout_ffn}. "
                f"Expected one of: {tuple(readout.ReadoutRegistry.values())}"
            )


def column_str_to_int(columns: list, header: list) -> list:
    if columns is None:
        return None
    if all(isinstance(col, str) for col in columns):
        columns = [i for i, name in enumerate(header) if name in columns]
    if not all(isinstance(col, int) for col in columns):
        raise ValueError("header and columns do not match")
    return columns
