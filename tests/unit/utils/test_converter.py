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
from chemprop.uncertainty.estimator import (
    EvidentialTotalEstimator,
    MVEEstimator,
    QuantileRegressionEstimator,
)
from chemprop.utils.v1_to_v2 import convert_model_file_v1_to_v2


@pytest.fixture
def example_model_v1_path(data_dir):
    return data_dir / "example_model_v1_regression_mol.pt"


@pytest.fixture
def example_model_v1_4_path(data_dir):
    return data_dir / "example_model_v1_4.pt"


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

    mpnn = MPNN.load_from_checkpoint(model_v2_save_path)

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


@pytest.mark.parametrize(
    ("method", "uncertainty_estimator"),
    [
        ("mve", MVEEstimator()),
        ("evidential", EvidentialTotalEstimator()),
        ("quantile", QuantileRegressionEstimator()),
    ],
)
def test_converter_with_uncertainty_method(data_dir, tmp_path, method, uncertainty_estimator):
    directory = tmp_path / "test_converter"
    directory.mkdir()
    model_v2_save_path = directory / f"example_model_v2_{method}_regression.npt"

    model_v1_path = data_dir / f"example_model_v1_{method}_regression.pt"
    convert_model_file_v1_to_v2(model_v1_path, model_v2_save_path, ignore_unsupported_metrics=True)
    assert model_v2_save_path.exists()

    path = data_dir / f"example_model_v1_{method}_regression_prediction.csv"

    with open(path, "r", encoding="utf-8") as fid:
        reader = csv.reader(fid)
        next(reader)
        smis, ys, uncs = zip(*[(smi, float(score), float(unc)) for smi, score, unc in reader])

    featurizer = SimpleMoleculeMolGraphFeaturizer(atom_featurizer=MultiHotAtomFeaturizer.v1())

    ys = np.array(ys).reshape(-1, 1)
    uncs = np.array(uncs).reshape(-1, 1)
    test_data = [MoleculeDatapoint.from_smi(smi, None) for smi in smis]
    test_dset = MoleculeDataset(test_data, featurizer)

    test_loader = build_dataloader(test_dset, shuffle=False)

    mpnn = MPNN.load_from_checkpoint(model_v2_save_path)
    trainer = pl.Trainer(accelerator="cpu", logger=None, enable_progress_bar=False)
    ys_v2, uncs_v2 = map(np.array, uncertainty_estimator(test_loader, [mpnn], trainer))

    assert np.allclose(ys_v2, ys, atol=1e-5)
    assert np.allclose(uncs_v2, uncs, atol=1e-5)
