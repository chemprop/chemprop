import pickle

import pytest
import torch
from torch import nn

from chemprop.nn import (
    MSE,
    BinaryAUROC,
    BondMessagePassing,
    ChempropMetric,
    EvidentialLoss,
    GraphTransform,
    MeanAggregation,
    QuantileLoss,
    R2Score,
    RegressionFFN,
    ScaleTransform,
)
from chemprop.nn.hparams import (
    class_path,
    from_hparams,
    get_init_args,
    get_native_chemprop_classes,
    get_native_torch_classes,
    resolve_class,
    serialize_hparams,
)


class CustomActivation(nn.Module):
    def forward(self, X):
        return X


def test_resolve_class_restricted():
    assert resolve_class("chemprop.nn.agg.MeanAggregation", True) is MeanAggregation
    assert resolve_class(MeanAggregation, True) is MeanAggregation

    for path in ["subprocess.Popen", "os.system", "builtins.eval", class_path(CustomActivation)]:
        with pytest.raises(pickle.UnpicklingError, match="add_safe_globals"):
            resolve_class(path, True)

    with torch.serialization.safe_globals([CustomActivation]):
        assert resolve_class(class_path(CustomActivation), True) is CustomActivation


def test_resolve_class_unrestricted():
    assert resolve_class(class_path(CustomActivation), False) is CustomActivation

    with pytest.raises(TypeError, match="not a class"):
        resolve_class("os.system", False)
    with pytest.raises(ImportError):
        resolve_class("chemprop.nn.agg.NotAClass", False)


def test_native_chemprop_classes():
    native_classes = get_native_chemprop_classes().values()
    for cls in [BondMessagePassing, MeanAggregation, RegressionFFN, MSE, R2Score, GraphTransform]:
        assert cls in native_classes
    assert ChempropMetric not in native_classes
    assert nn.Identity not in native_classes
    assert nn.ReLU not in native_classes


def test_native_torch_classes():
    native_classes = get_native_torch_classes().values()
    for cls in [nn.Identity, nn.ReLU, nn.LeakyReLU]:
        assert cls in native_classes
    assert MSE not in native_classes


@pytest.mark.parametrize(
    "metric, expected",
    [
        (MSE(task_weights=[0.5, 2.0]), {"task_weights": [0.5, 2.0]}),
        (EvidentialLoss(v_kl=0.3), {"task_weights": 1.0, "v_kl": 0.3, "eps": 1e-8}),
        (QuantileLoss(alpha=0.2), {"task_weights": 1.0, "alpha": 0.2}),
        (R2Score(), {"task_weights": 1.0}),
        (BinaryAUROC(), {"task_weights": 1.0}),
    ],
)
def test_serialize_and_build_metric(metric, expected):
    serialized = serialize_hparams(metric)
    assert serialized == {"cls": class_path(type(metric)), **expected}

    built = from_hparams(serialized, restrict_classes=True)
    assert type(built) is type(metric)
    assert dict(built.hparams) == dict(metric.hparams)
    assert torch.equal(built.task_weights, metric.task_weights)


def test_metric_hparams_exclude_outer_init_args():
    """a metric built in another object's ``__init__`` doesn't save that object's arguments"""

    class Outer(nn.Module):
        def __init__(self, n_layers: int = 3, v_kl: float = 0.5):
            super().__init__()
            self.metric = EvidentialLoss(v_kl=0.3)

    assert Outer().metric.hparams == {
        "cls": EvidentialLoss,
        "task_weights": 1.0,
        "v_kl": 0.3,
        "eps": 1e-8,
    }


def test_get_init_args_with_kwargs():
    class Base(nn.Module):
        def __init__(self, a: int = 1, **kwargs):
            super().__init__()
            self.init_args = get_init_args(self)

    class Child(Base):
        def __init__(self, a: int = 2, **kwargs):
            super().__init__(a, **kwargs)

    assert Child(b=3).init_args == {"a": 2, "b": 3}


def test_serialize_local_metric(caplog):
    """a class defined in a function can't be imported from its path, so it is saved whole"""

    class LocalMetric(MSE):
        pass

    metric = LocalMetric()
    serialized = serialize_hparams(metric)

    assert serialized["cls"] is LocalMetric
    assert "weights_only=False" in caplog.text


def test_serialize_and_build_hparams():
    hparams = {"transform": ScaleTransform(mean=[1.0], scale=[2.0]), "values": (1, [nn.ReLU()])}
    serialized = serialize_hparams(hparams)

    assert serialized["transform"]["cls"] == "chemprop.nn.transforms.ScaleTransform"
    assert serialized["values"] == (
        1,
        [{"cls": "torch.nn.modules.activation.ReLU", "inplace": False}],
    )

    built = from_hparams(serialized, restrict_classes=True)
    assert isinstance(built["transform"], ScaleTransform)
    assert torch.equal(built["transform"].scale, torch.tensor([[2.0]]))
    assert isinstance(built["values"][1][0], nn.ReLU)


def test_from_hparams_with_class():
    """hyperparameters saved before Chemprop v2.4 store the class itself"""
    agg = from_hparams({"cls": MeanAggregation, "dim": 0}, restrict_classes=True)

    assert isinstance(agg, MeanAggregation)
