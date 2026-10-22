from collections.abc import Mapping
from enum import Enum
import inspect
import logging
import pickle
import pydoc
from typing import Any, Protocol, TypedDict

import numpy as np
import torch
from torch import Tensor, nn

logger = logging.getLogger(__name__)


class HParamsDict(TypedDict):
    """A dictionary containing a module's class and its hyperparameters

    The values may contain the hyperparameters of nested modules as dictionaries of this form. A
    module can be rebuilt from its hyperparameters with :func:`from_hparams`::

        module = from_hparams(hparams, restrict_classes=False)
    """

    cls: type


class HasHParams(Protocol):
    """A protocol for classes with an :attr:`hparams` attribute containing the object's class and
    the arguments required to initialize it.

    That is, any object which implements :class:`HasHParams` should be able to be rebuilt via::

        class Foo(HasHParams):
            def __init__(self, *args, **kwargs):
                ...

        foo1 = Foo(...)
        foo2 = from_hparams(foo1.hparams, restrict_classes=False)
        # code to compare foo1 and foo2 goes here and they should be equal
    """

    hparams: HParamsDict


def class_path(cls: type) -> str:
    """Get the fully qualified path of a class, e.g., ``"chemprop.nn.agg.MeanAggregation"``"""
    return f"{cls.__module__}.{cls.__qualname__}"


def get_native_chemprop_classes() -> dict[str, type]:
    """Get the classes that Chemprop natively allows :func:`from_hparams` to build with
    ``restrict_classes=True``.
    """
    # imported here because the modules in chemprop.nn import this module
    from chemprop.nn import (
        AggregationRegistry,
        AtomMessagePassing,
        AttentiveAggregation,
        BondMessagePassing,
        ConstrainerFFN,
        GraphTransform,
        LossFunctionRegistry,
        MABAtomMessagePassing,
        MABBondMessagePassing,
        MetricRegistry,
        MulticomponentMessagePassing,
        PredictorRegistry,
        ScaleTransform,
        UnscaleTransform,
    )

    classes = [
        *AggregationRegistry.values(),
        *PredictorRegistry.values(),
        *MetricRegistry.values(),
        *LossFunctionRegistry.values(),
        AttentiveAggregation,
        AtomMessagePassing,
        BondMessagePassing,
        MABAtomMessagePassing,
        MABBondMessagePassing,
        MulticomponentMessagePassing,
        ConstrainerFFN,
        ScaleTransform,
        UnscaleTransform,
        GraphTransform,
    ]

    return {class_path(cls): cls for cls in classes}


def get_native_torch_classes() -> dict[str, type]:
    classes = [
        nn.Identity,
        *(getattr(nn.modules.activation, name) for name in nn.modules.activation.__all__),
    ]

    return {class_path(cls): cls for cls in classes}


def get_user_safe_classes() -> dict[str, type]:
    if not hasattr(torch.serialization, "get_safe_globals"):
        return {}

    classes = {}
    for obj in torch.serialization.get_safe_globals():
        if isinstance(obj, tuple):
            obj, path = obj
        else:
            path = None
        if not isinstance(obj, type):
            continue
        classes[path or class_path(obj)] = obj

    return classes


def resolve_class(cls: str | type, restrict_classes: bool) -> type:
    """Get the actual class corresponding to a ``"cls"`` hparams entry.

    Parameters
    ----------
    cls : str | type
        the class path (e.g., ``"chemprop.nn.agg.MeanAggregation"``) or the class itself (for
        pre-v2.4 model files)
    restrict_classes : bool
        if ``True``, only classes returned by :func:`get_native_chemprop_classes` or
        :func:`get_native_torch_classes` or allowed with
        :func:`torch.serialization.add_safe_globals` can be resolved. If ``False``, the class is
        imported from its path.

    Raises
    ------
    pickle.UnpicklingError
        if ``restrict_classes`` is ``True`` and the class is not allowed
    """
    if isinstance(cls, type):
        return cls
    if not isinstance(cls, str):
        raise TypeError(f"Expected a class or a class path! got: {type(cls)}")

    if restrict_classes:
        resolved = (
            get_native_chemprop_classes().get(cls)
            or get_native_torch_classes().get(cls)
            or get_user_safe_classes().get(cls)
        )
        if resolved is None:
            raise pickle.UnpicklingError(
                f"Class '{cls}' is not allowed when loading with `weights_only=True`. If you trust "
                "this class, allow it with `torch.serialization.add_safe_globals([YourClass])` "
                "before loading the model."
            )
    else:
        resolved = pydoc.locate(cls)
        if resolved is None:
            raise ImportError(f"Could not import class '{cls}'")

    if not isinstance(resolved, type):
        raise TypeError(f"'{cls}' is not a class! got: {type(resolved)}")

    return resolved


def serialize_hparams(obj: Any) -> Any:
    """Recursively convert hyperparameters into a form that can be loaded with
    ``torch.load(..., weights_only=True)``. Objects that can't be converted are left as is and a
    warning is logged, as the model file will need to be loaded with ``weights_only=False``.
    """
    match obj:
        case Enum():
            return obj.value
        case np.generic():
            return obj.item()
        case np.ndarray():
            return torch.from_numpy(obj)
        case None | str() | int() | float() | bool() | Tensor():
            return obj
        case type() if pydoc.locate(class_path(obj)) is obj:
            return class_path(obj)
        case Mapping():
            return {k: serialize_hparams(v) for k, v in obj.items()}
        case list():
            return [serialize_hparams(v) for v in obj]
        case tuple():
            return tuple(serialize_hparams(v) for v in obj)
        case nn.Module(hparams=Mapping() as hparams) if "cls" in hparams:
            return serialize_hparams(hparams)
        case nn.Module() if class_path(type(obj)) in get_native_torch_classes():
            # The torch classes don't save hparams, but they do use the same name for their
            # ``__init__`` args as for their attributes, e.g., ``nn.LeakyReLU(0.2)`` sets
            # ``self.negative_slope = 0.2``, so they can be rebuilt from those attributes.
            params = inspect.signature(type(obj)).parameters
            kwargs = {name: getattr(obj, name) for name in params if hasattr(obj, name)}
            return {"cls": class_path(type(obj)), **serialize_hparams(kwargs)}

    logger.warning(
        f"Could not convert {obj!r} to a format that can be loaded with `weights_only=True`. "
        "The object itself will be saved instead, so the saved model must be loaded with "
        "`weights_only=False`."
    )
    return obj


def from_hparams(hparams: Any, restrict_classes: bool) -> Any:
    """Recursively build objects from hyperparameters.

    Parameters
    ----------
    hparams : Any
        the hyperparameters to build objects from
    restrict_classes : bool
        if ``True``, only allowed classes can be built. See :func:`resolve_class`. This should be
        ``True`` whenever the model file was loaded with ``weights_only=True``.
    """
    if isinstance(hparams, Mapping):
        if "cls" in hparams:
            cls = resolve_class(hparams["cls"], restrict_classes)
            kwargs = {
                k: from_hparams(v, restrict_classes) for k, v in hparams.items() if k != "cls"
            }
            return cls(**kwargs)
        return {k: from_hparams(v, restrict_classes) for k, v in hparams.items()}
    if isinstance(hparams, list):
        return [from_hparams(v, restrict_classes) for v in hparams]
    if isinstance(hparams, tuple):
        return tuple(from_hparams(v, restrict_classes) for v in hparams)

    return hparams


def get_init_args(obj: Any) -> dict[str, Any]:
    """Get the arguments that ``obj`` was constructed with. This must be called from ``__init__``.

    When ``obj`` is being constructed, the call stack contains one ``__init__`` call for each
    class in its hierarchy that calls ``super().__init__(...)``.
    For example, ``EvidentialLoss(v_kl=0.3)`` gives this stack:

    .. code-block:: text

        ChempropMetric.__init__(self, task_weights=1.0)           <- caller of get_init_args
        EvidentialLoss.__init__(self, task_weights=1.0, v_kl=0.3, eps=1e-8)
        <code that called EvidentialLoss(v_kl=0.3)>               <- not an __init__ of obj

    This walks up the stack from the caller, merging the arguments of each ``__init__`` call, and
    stops at the first call that isn't an ``__init__`` of ``obj``. Here, it returns
    ``{"task_weights": 1.0, "v_kl": 0.3, "eps": 1e-8}``. If an argument appears in several calls,
    the outermost value wins, as that's the one the user passed to the subclass. The contents of
    ``**kwargs`` are included as separate arguments, and ``*args`` are ignored. Values are read when
    this is called, so a subclass that reassigns an argument before calling ``super().__init__``
    saves the reassigned value.

    Lightning's ``save_hyperparameters`` walks the stack in the same way, but its stopping rule
    (in ``lightning.pytorch.utilities.parsing.collect_init_args``) is::

        if "__class__" in local_vars and (not classes or isinstance(local_self, classes)):
            path_args.append(local_args)
            return collect_init_args(frame.f_back, path_args, inside=True, classes=classes)

    with ``classes=(HyperparametersMixin,)``. That only checks that a call is an ``__init__`` of
    *some* ``HyperparametersMixin``, not of ``obj``. So if ``obj`` is built inside the ``__init__``
    of a model, it keeps walking into the model's ``__init__`` and saves the model's arguments too.
    For example, ``MPNN.__init__`` builds its default metric, which then saved ``message_passing``,
    ``agg``, ``predictor``, etc. as its own hyperparameters. Checking that ``self`` is ``obj``
    stops before that frame.
    """
    init_args = {}
    # start from the `__init__` that called this function
    frame = inspect.currentframe().f_back
    while (
        frame is not None
        and frame.f_code.co_name == "__init__"
        # the `__init__` is running for `obj`, not for an object that is building `obj`
        and frame.f_locals.get("self") is obj
    ):
        info = inspect.getargvalues(frame)
        frame_args = {name: info.locals[name] for name in info.args if name != "self"}
        if info.keywords is not None:
            frame_args |= info.locals[info.keywords]
        init_args |= frame_args

        frame = frame.f_back

    return init_args
