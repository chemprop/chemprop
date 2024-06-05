import csv

from lightning import pytorch as pl
import numpy as np
import pytest

from chemprop.data.dataloader import build_dataloader
from chemprop.data.datapoints import MoleculeDatapoint
from chemprop.data.datasets import MoleculeDataset
from chemprop.featurizers.atom import MultiHotAtomFeaturizer
from chemprop.featurizers.molgraph.molecule import SimpleMoleculeMolGraphFeaturizer
from chemprop.models.model import MPNN
from chemprop.utils.v1_to_v2 import convert_model_file_v1_to_v2


@pytest.fixture
def example_model_v1_path(data_dir):
    return data_dir / "example_model_v1_regression_mol.pt"


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
    model_v2_save_path = directory / "example_model_v2_regression_mol.ckpt"

    convert_model_file_v1_to_v2(example_model_v1_path, model_v2_save_path)
    assert model_v2_save_path.exists()

    mpnn = MPNN.load_from_checkpoint(model_v2_save_path)

    ys_v1, test_loader = example_model_v1_prediction

    trainer = pl.Trainer(accelerator="cpu", logger=None, enable_progress_bar=False)
    predss = trainer.predict(mpnn, test_loader)
    ys_v2 = np.vstack(predss)
    assert np.allclose(ys_v2, ys_v1, atol=1e-6)
