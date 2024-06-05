from pathlib import Path

from lightning import pytorch as pl
import numpy as np
import pytest
from torch.utils.data import DataLoader

from chemprop.data import MoleculeDatapoint, MoleculeDataset, collate_batch
from chemprop.models import MPNN
from chemprop.models.utils import save_model


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
