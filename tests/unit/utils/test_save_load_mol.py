from pathlib import Path

from lightning import pytorch as pl
import numpy as np
import pytest
import torch
from torch.nn import Identity
from torch.utils.data import DataLoader

from chemprop.data import MoleculeDatapoint, MoleculeDataset, collate_batch
from chemprop.models import MPNN
from chemprop.models.utils import load_model, save_model
from chemprop.nn import (
    MSE,
    BondMessagePassing,
    GraphTransform,
    NormAggregation,
    RegressionFFN,
    ScaleTransform,
    UnscaleTransform,
)


@pytest.fixture
def checkpoint_path(data_dir):
    return data_dir / "example_model_v2_regression_mol.ckpt"


@pytest.fixture
def model_path(data_dir):
    return data_dir / "example_model_v2_regression_mol.pt"


@pytest.fixture
def model(checkpoint_path):
    model = MPNN.load_from_checkpoint(checkpoint_path)
    return model


@pytest.fixture
def test_loader(smis):
    data = [MoleculeDatapoint.from_smi(smi) for smi in smis]
    dset = MoleculeDataset(data)

    return DataLoader(dset, 32, collate_fn=collate_batch)


@pytest.fixture
def trainer():
    return pl.Trainer(
        logger=None,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        accelerator="cpu",
        devices=1,
    )


@pytest.fixture
def ys(model, test_loader, trainer):
    predss = trainer.predict(model, test_loader)
    return np.vstack(predss)


def test_roundtrip(tmp_path, model, test_loader, trainer, ys):
    save_path = Path(tmp_path) / "test.pt"
    save_model(save_path, model)

    model_from_file = MPNN.load_from_file(save_path)

    predss_from_file = trainer.predict(model_from_file, test_loader)
    ys_from_file = np.vstack(predss_from_file)

    assert np.allclose(ys_from_file, ys, atol=1e-6)


def test_checkpoint_is_valid(checkpoint_path, test_loader, trainer, ys):
    model_from_checkpoint = MPNN.load_from_file(checkpoint_path)

    predss_from_checkpoint = trainer.predict(model_from_checkpoint, test_loader)
    ys_from_checkpoint = np.vstack(predss_from_checkpoint)

    assert np.allclose(ys_from_checkpoint, ys, atol=1e-6)


def test_checkpoint_roundtrip(checkpoint_path, model_path, trainer, test_loader):
    model_from_checkpoint = MPNN.load_from_checkpoint(checkpoint_path, map_location="cpu")
    model_from_file = MPNN.load_from_file(model_path, map_location="cpu")

    predss_from_checkpoint = trainer.predict(model_from_checkpoint, test_loader)
    ys_from_checkpoint = np.vstack(predss_from_checkpoint)

    predss_from_file = trainer.predict(model_from_file, test_loader)
    ys_from_file = np.vstack(predss_from_file)

    assert np.allclose(ys_from_file, ys_from_checkpoint, atol=1e-6)


def test_scalers_roundtrip(tmp_path):
    E_f_transform = ScaleTransform(mean=[0.0, 1.0], scale=[2.0, 3.0])
    graph_transform = GraphTransform(V_transform=Identity(), E_transform=E_f_transform)
    V_d_transform = ScaleTransform(mean=[4.0, 5.0], scale=[6.0, 7.0])
    mp = BondMessagePassing(graph_transform=graph_transform, V_d_transform=V_d_transform)

    output_transform = UnscaleTransform(mean=[8.0, 9.0], scale=[10.0, 11.0])
    criterion = MSE(task_weights=[12.0])
    ffn = RegressionFFN(output_transform=output_transform, criterion=criterion)

    X_d_transform = ScaleTransform(mean=[13.0, 14.0], scale=[15.0, 16.0])
    original = MPNN(mp, NormAggregation(), ffn, X_d_transform=X_d_transform)

    save_model(tmp_path / "model.pt", original)
    loaded = load_model(tmp_path / "model.pt", multicomponent=False)

    assert torch.equal(
        original.message_passing.V_d_transform.mean, loaded.message_passing.V_d_transform.mean
    )
    assert torch.equal(
        original.message_passing.graph_transform.E_transform.mean,
        loaded.message_passing.graph_transform.E_transform.mean,
    )
    assert torch.equal(
        original.predictor.criterion.task_weights, loaded.predictor.criterion.task_weights
    )
    assert torch.equal(
        original.predictor.output_transform.mean, loaded.predictor.output_transform.mean
    )
    assert torch.equal(original.X_d_transform.mean, loaded.X_d_transform.mean)


def test_load_checkpoint_with_metrics(data_dir):
    MPNN.load_from_checkpoint(data_dir / "example_model_v2_regression_mol_with_metrics.ckpt")
    MPNN.load_from_checkpoint(data_dir / "example_model_v2_classification_mol_with_metrics.ckpt")


def test_load_trained_on_cuda(data_dir):
    MPNN.load_from_file(data_dir / "example_model_v2_trained_on_cuda.pt", map_location="cpu")
