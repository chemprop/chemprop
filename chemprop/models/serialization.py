from collections.abc import Mapping
import inspect
import io
from os import PathLike
import pickle
from typing import IO, Any, Self
import warnings

from lightning import pytorch as pl
import torch

from chemprop.nn.hparams import from_hparams, get_native_chemprop_classes, serialize_hparams
from chemprop.nn.transforms import GraphTransform, _ScaleTransformMixin
from chemprop.utils.registry import Factory

WEIGHTS_ONLY_DEFAULT_WARNING = (
    "`weights_only` was not specified, so it defaults to `False`, which allows arbitrary objects "
    "to be loaded from the model file. This can execute arbitrary code, so only load model files "
    "you trust. In a future release of Chemprop, the default will change to `True`. Pass "
    "`weights_only` explicitly (or `--weights-only true/false` on the command line) to silence this warning."
)
LIGHTNING_SUPPORTS_WEIGHTS_ONLY = (
    "weights_only" in inspect.signature(pl.LightningModule.load_from_checkpoint).parameters
)


def resolve_weights_only(weights_only: bool | None) -> bool:
    """Get the value of ``weights_only`` to use, warning that the default will change if it is
    ``None``"""
    if weights_only is None:
        warnings.warn(WEIGHTS_ONLY_DEFAULT_WARNING, FutureWarning, stacklevel=3)
        return False

    return weights_only


def load_checkpoint(
    path: PathLike | IO, map_location: Any = None, weights_only: bool | None = None
) -> dict:
    """Load a model file or checkpoint into a dictionary.

    Parameters
    ----------
    path : PathLike | IO
        the path to the model file or checkpoint
    map_location : Any, default=None
        passed to :func:`torch.load`
    weights_only : bool | None, default=None
        passed to :func:`torch.load`. If ``None``, ``False`` is used with a warning that the default
        will change to ``True`` in a future release.

    Raises
    ------
    pickle.UnpicklingError
        if ``weights_only`` is ``True`` and the file contains objects other than tensors and
        primitive types. This happens for model files saved before Chemprop v2.4.
    """
    weights_only = resolve_weights_only(weights_only)

    try:
        return torch.load(path, map_location=map_location, weights_only=weights_only)
    except pickle.UnpicklingError as e:
        if weights_only:
            raise pickle.UnpicklingError(
                f"Could not load '{path}' with `weights_only=True`. This model file was likely "
                "saved before Chemprop v2.4, which saved entire objects. If you trust the source "
                "of this file, either load it with `weights_only=False` (`--weights-only false` on "
                "the command line) or convert it to the v2.4 format once with "
                f"`chemprop convert -i {path}`."
            ) from e
        raise


def needs_metric_rebuild(metric: Any) -> bool:
    if metric is None or isinstance(metric, Mapping):
        return False
    if type(metric) in get_native_chemprop_classes().values() and "hparams" not in vars(metric):
        return True
    if not hasattr(metric, "_defaults"):
        return True
    if not torch.cuda.is_available() and metric.device.type != "cpu":
        return True

    return False


def rebuild_metric(metric: Any) -> Any:
    return Factory.build(metric.__class__, task_weights=metric.task_weights, **metric.__dict__)


def rebuild_saved_metrics(hparams: dict) -> None:
    """Rebuild the metrics and predictor criteria in ``hparams`` that need it (see
    :func:`needs_metric_rebuild`), in place"""
    if hparams.get("metrics") is not None:
        hparams["metrics"] = [
            rebuild_metric(metric) if needs_metric_rebuild(metric) else metric
            for metric in hparams["metrics"]
        ]

    for key in ("predictor", "mol_predictor", "atom_predictor", "bond_predictor"):
        predictor = hparams.get(key)
        if isinstance(predictor, Mapping) and needs_metric_rebuild(predictor.get("criterion")):
            predictor["criterion"] = rebuild_metric(predictor["criterion"])


def rebuild_transform(obj: Any) -> Any:
    """Recursively rebuild the transforms in ``obj`` that were unpickled from pre-v2.4 model files"""
    match obj:
        case Mapping():
            return {k: rebuild_transform(v) for k, v in obj.items()}
        case list():
            return [rebuild_transform(v) for v in obj]
        case tuple():
            return tuple(rebuild_transform(v) for v in obj)
        case _ScaleTransformMixin() if "hparams" not in vars(obj):
            return type(obj)(obj.mean[0].cpu(), obj.scale[0].cpu())
        case GraphTransform() if "hparams" not in vars(obj):
            return GraphTransform(
                rebuild_transform(obj.V_transform), rebuild_transform(obj.E_transform)
            )

    return obj


def rebuild_saved_transforms(hparams: dict) -> None:
    hparams.update(rebuild_transform(hparams))


class CheckpointMixin:
    """Saving and loading shared by Chemprop's models. It must come before
    :class:`lightning.pytorch.LightningModule` in the base classes."""

    def on_save_checkpoint(self, checkpoint: dict) -> None:
        if self.CHECKPOINT_HYPER_PARAMS_KEY in checkpoint:
            checkpoint[self.CHECKPOINT_HYPER_PARAMS_KEY] = serialize_hparams(
                checkpoint[self.CHECKPOINT_HYPER_PARAMS_KEY]
            )

    @classmethod
    def _load(
        cls, path: PathLike | IO, map_location: Any, weights_only: bool, **kwargs
    ) -> tuple[dict, dict, dict]:
        """This function primarily builds the submodules that the parent module needs. It also
        provides some helpers to maintain backwards compatibility."""
        d = load_checkpoint(path, map_location, weights_only)

        try:
            hparams = d["hyper_parameters"]
            state_dict = d["state_dict"]
        except KeyError:
            raise KeyError(f"Could not find hyper parameters and/or state dict in {path}.")

        rebuild_saved_metrics(hparams)
        rebuild_saved_transforms(hparams)

        hparams = {
            key: from_hparams(value, restrict_classes=weights_only)
            for key, value in hparams.items()
            if key not in kwargs
        }

        return hparams | kwargs, state_dict, d

    @classmethod
    def load_from_checkpoint(
        cls,
        checkpoint_path,
        map_location=None,
        hparams_file=None,
        strict=True,
        weights_only: bool | None = None,
        **kwargs,
    ) -> Self:
        weights_only = resolve_weights_only(weights_only)
        hparams, state_dict, checkpoint = cls._load(
            checkpoint_path, map_location, weights_only, **kwargs
        )
        checkpoint["state_dict"] = state_dict
        checkpoint["hyper_parameters"] = {}
        # model files from `save_model` don't have this key, but Lightning requires it
        checkpoint.setdefault("pytorch-lightning_version", pl.__version__)
        buffer = io.BytesIO()
        torch.save(checkpoint, buffer)
        buffer.seek(0)

        if LIGHTNING_SUPPORTS_WEIGHTS_ONLY:
            return super().load_from_checkpoint(
                buffer, map_location, hparams_file, strict, weights_only=weights_only, **hparams
            )

        try:
            return super().load_from_checkpoint(
                buffer, map_location, hparams_file, strict, **hparams
            )
        except pickle.UnpicklingError as e:
            raise pickle.UnpicklingError(
                "The checkpoint contains objects that can only be loaded with "
                f"`weights_only=False`, but Lightning v{pl.__version__} doesn't support passing "
                "`weights_only`, and PyTorch v2.6+ defaults to `weights_only=True`. Either upgrade "
                "to `lightning>=2.6`, downgrade to `torch<2.6`, or convert the model file to the "
                "latest chemprop format with `chemprop convert -i <model_path>`."
            ) from e

    @classmethod
    def load_from_file(
        cls, model_path, map_location=None, strict=True, weights_only: bool | None = None, **kwargs
    ) -> Self:
        """Equivalent to :meth:`load_from_checkpoint`, kept for backwards compatibility"""
        return cls.load_from_checkpoint(
            model_path, map_location, strict=strict, weights_only=weights_only, **kwargs
        )
