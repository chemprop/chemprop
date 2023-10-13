import csv
import pytest
import numpy as np

from pathlib import Path

from lightning import pytorch as pl

from chemprop.v2 import data
from chemprop.v2 import featurizers
from chemprop.v2.models.model import MPNN
from chemprop.v2.utils.v1_to_v2 import convert_model_file_v1_to_v2

TEST_DIR = Path(__file__).parents[2]
TEST_DATA_DIR = TEST_DIR / "data"


@pytest.fixture
def example_model_v1_path():
    return TEST_DATA_DIR / "example_model_v1.pt"


@pytest.fixture
def example_model_v1_prediction():
    path = TEST_DATA_DIR / "example_model_v1_prediction.csv"

    with open(path) as fid:
        reader = csv.reader(fid)
        next(reader)
        smis, ys = zip(*[(smi, float(score)) for smi, score in reader])

    featurizer = featurizers.MoleculeMolGraphFeaturizer()

    ys = np.array(ys).reshape(-1, 1)
    test_data = [data.MoleculeDatapoint.from_smi(smi, None) for smi in smis]
    test_dset = data.MoleculeDataset(test_data, featurizer)

    test_loader = data.MolGraphDataLoader(test_dset)
    return ys, test_loader


def test_converter(tmp_path, example_model_v1_path, example_model_v1_prediction):
    directory = tmp_path / "test_converter"
    directory.mkdir()
    model_v2_save_path = directory / "example_model_v2.ckpt"

    convert_model_file_v1_to_v2(example_model_v1_path, model_v2_save_path)
    assert model_v2_save_path.exists()

    mpnn = MPNN.load_from_checkpoint(model_v2_save_path)

    ys_v1, test_loader = example_model_v1_prediction

    trainer = pl.Trainer(logger=None, enable_progress_bar=False)
    predss = trainer.predict(mpnn, test_loader)
    ys_v2 = np.vstack(predss)
    assert np.allclose(ys_v2, ys_v1, atol=1e-6)
