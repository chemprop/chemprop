import csv

from lightning import pytorch as pl
import numpy as np
import pytest
import torch

from chemprop.data.dataloader import build_dataloader
from chemprop.data.datapoints import MoleculeDatapoint
from chemprop.data.datasets import MoleculeDataset
from chemprop.featurizers.atom import MultiHotAtomFeaturizer
from chemprop.featurizers.molgraph.molecule import SimpleMoleculeMolGraphFeaturizer
from chemprop.models.model import MPNN
from chemprop.utils.v1_to_v2 import convert_model_file_v1_to_v2
from chemprop.utils.v2_1_to_v2_4 import convert_model_file_to_latest

# these tests load v1 model files with `weights_only=False`
pytestmark = pytest.mark.unsafe_load


@pytest.fixture
def example_model_v1_path(pre_v2_4_model_path):
    return pre_v2_4_model_path("v1_model_files/example_model_v1_regression_mol.pt")


@pytest.fixture
def example_model_v1_4_path(pre_v2_4_model_path):
    return pre_v2_4_model_path("v1_model_files/example_model_v1_4.pt")


@pytest.fixture
def example_model_v1_prediction(data_dir):
    path = data_dir / "example_model_v1_regression_mol_prediction.csv"

    with open(path) as fid:
        reader = csv.reader(fid)
        next(reader)
        smis, ys = zip(*[(smi, float(score)) for smi, score in reader])

    featurizer = SimpleMoleculeMolGraphFeaturizer(atom_featurizer=MultiHotAtomFeaturizer.v1())

    ys = np.array(ys).reshape(-1, 1)
    test_data = [MoleculeDatapoint.from_smi(smi, None) for smi in smis]
    test_dset = MoleculeDataset(test_data, featurizer)

    test_loader = build_dataloader(test_dset, shuffle=False)
    return ys, test_loader


def test_converter(tmp_path, example_model_v1_path, example_model_v1_prediction):
    directory = tmp_path / "test_converter"
    directory.mkdir()
    model_v2_save_path = directory / "example_model_v2_regression_mol.pt"

    convert_model_file_v1_to_v2(example_model_v1_path, model_v2_save_path)
    assert model_v2_save_path.exists()

    # the v1 to v2 conversion keeps arbitrary objects, so the converted file isn't weights_only safe
    mpnn = MPNN.load_from_checkpoint(model_v2_save_path, weights_only=False)

    ys_v1, test_loader = example_model_v1_prediction

    trainer = pl.Trainer(accelerator="cpu", logger=None, enable_progress_bar=False)
    predss = trainer.predict(mpnn, test_loader)
    ys_v2 = np.vstack(predss)
    assert np.allclose(ys_v2, ys_v1, atol=1e-6)


def test_converter_v1_4(tmp_path, example_model_v1_4_path):
    directory = tmp_path / "test_converter"
    directory.mkdir()
    model_v2_save_path = directory / "converted_v1_4.pt"

    convert_model_file_v1_to_v2(example_model_v1_4_path, model_v2_save_path)
    assert model_v2_save_path.exists()


def test_convert_v1_to_latest(tmp_path, example_model_v1_path, example_model_v1_prediction):
    save_path = tmp_path / "example_model_latest.pt"

    convert_model_file_to_latest(example_model_v1_path, save_path)
    assert torch.serialization.get_unsafe_globals_in_checkpoint(save_path) == []

    mpnn = MPNN.load_from_checkpoint(save_path, weights_only=True)

    ys_v1, test_loader = example_model_v1_prediction

    trainer = pl.Trainer(accelerator="cpu", logger=None, enable_progress_bar=False)
    predss = trainer.predict(mpnn, test_loader)
    ys_v2 = np.vstack(predss)
    assert np.allclose(ys_v2, ys_v1, atol=1e-6)


def test_convert_v1_4_to_latest(tmp_path, example_model_v1_4_path):
    save_path = tmp_path / "converted_v1_4_latest.pt"

    convert_model_file_to_latest(example_model_v1_4_path, save_path)

    assert torch.serialization.get_unsafe_globals_in_checkpoint(save_path) == []
