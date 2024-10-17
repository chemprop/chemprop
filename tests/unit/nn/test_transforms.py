import numpy as np
import pytest
from sklearn.preprocessing import StandardScaler
import torch

from chemprop.nn.transforms import GraphTransform, ScaleTransform, UnscaleTransform


class MockBatchMolGraph:
    def __init__(self, V, E):
        self.V = V
        self.E = E


@pytest.fixture
def mean():
    return np.array([0.0, 1.0], dtype=np.float32)


@pytest.fixture
def scale():
    return np.array([2.0, 3.0], dtype=np.float32)


@pytest.fixture
def pad():
    return 2


@pytest.fixture
def tensor_mean(mean, pad):
    return torch.cat([torch.zeros(pad), torch.tensor(mean, dtype=torch.float)])


@pytest.fixture
def tensor_scale(scale, pad):
    return torch.cat([torch.ones(pad), torch.tensor(scale, dtype=torch.float)])


@pytest.fixture
def scaler(mean, scale):
    scaler = StandardScaler()
    scaler.mean_ = mean
    scaler.scale_ = scale
    return scaler


@pytest.fixture
def scale_transform(scaler, pad):
    return ScaleTransform.from_standard_scaler(scaler, pad)


@pytest.fixture
def unscale_transform(scaler, pad):
    return UnscaleTransform.from_standard_scaler(scaler, pad=0)


@pytest.fixture
def graph_transform(scale_transform):
    return GraphTransform(V_transform=scale_transform, E_transform=scale_transform)


@pytest.fixture
def X():
    return torch.tensor([[99.0, 99.0, 1.0, 2.0], [99.0, 99.0, 3.0, 4.0]])


@pytest.fixture
def prediction():
    return torch.tensor([[1.0, 2.0]])


@pytest.fixture
def variance():
    return torch.tensor([[0.1, 0.2]])


@pytest.fixture
def bmg():
    V = torch.tensor([[99.0, 99.0, 1.0, 2.0], [99.0, 99.0, 3.0, 4.0]])
    E = torch.tensor([[99.0, 99.0, 1.0, 2.0], [99.0, 99.0, 3.0, 4.0]])
    return MockBatchMolGraph(V=V, E=E)


def test_uneven_shapes():
    with pytest.raises(ValueError):
        ScaleTransform(mean=[0.0], scale=[1.0, 2.0])


def test_padding(mean, scale, pad):
    scale_transform = ScaleTransform(mean, scale, pad)
    assert torch.all(scale_transform.mean[0, :pad] == 0.0).item()
    assert torch.all(scale_transform.scale[0, :pad] == 1.0).item()


def test_from_standard_scaler(mean, scale, scaler):
    scale_transform = ScaleTransform.from_standard_scaler(scaler)

    assert torch.all(scale_transform.mean == torch.tensor([0.0, 1.0])).item()
    assert torch.all(scale_transform.scale == torch.tensor([2.0, 3.0])).item()


def test_scale_transform_forward_train(scale_transform, X):
    scale_transform.train()
    output_X = scale_transform(X)
    assert output_X is X


def test_scale_transform_forward_eval(tensor_mean, tensor_scale, scale_transform, X):
    scale_transform.eval()
    output_X = scale_transform(X)
    expected_X = (X - tensor_mean) / tensor_scale
    assert torch.equal(output_X, expected_X)


def test_unscale_transform_forward_train(unscale_transform, X):
    unscale_transform.train()
    output_X = unscale_transform(X)
    assert output_X is X


def test_unscale_transform_forward_eval(mean, scale, unscale_transform, prediction):
    unscale_transform.eval()
    output = unscale_transform(prediction)
    expected = prediction * scale + mean
    assert torch.equal(output, expected)


def test_unscale_transform_variance_train(unscale_transform, variance):
    unscale_transform.train()
    output_variance = unscale_transform.transform_variance(variance)
    assert output_variance is variance


def test_unscale_transform_variance_eval(scale, unscale_transform, variance):
    unscale_transform.eval()
    output_variance = unscale_transform.transform_variance(variance)
    expected_variance = variance * scale**2
    assert torch.equal(output_variance, expected_variance)


def test_graph_transform_forward_train(graph_transform, bmg):
    graph_transform.train()
    output_bmg = graph_transform(bmg)
    assert output_bmg is bmg


def test_graph_transform_forward_eval(graph_transform, bmg):
    graph_transform.eval()
    expected_V = graph_transform.V_transform(bmg.V)
    expected_E = graph_transform.E_transform(bmg.E)

    transformed_bmg = graph_transform(bmg)

    assert torch.equal(transformed_bmg.V, expected_V)
    assert torch.equal(transformed_bmg.E, expected_E)
