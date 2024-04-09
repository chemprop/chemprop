"""This tests the CLI functionality of training and predicting a regression model on a multi-molecule.
"""

import pytest

from chemprop.cli.main import main

pytestmark = pytest.mark.CLI


@pytest.fixture
def data_path(data_dir):
    return (
        str(data_dir / "regression" / "mol+mol" / "mol+mol.csv"), 
        str(data_dir / "regression" / "mol+mol" / "descriptors.npz"),
        ("0", str(data_dir / "regression" / "mol+mol" / "atom_features_0.npz")),
        ("1", str(data_dir / "regression" / "mol+mol" / "atom_features_1.npz")),
        ("0", str(data_dir / "regression" / "mol+mol" / "bond_features_0.npz")),
        ("1", str(data_dir / "regression" / "mol+mol" / "atom_descriptors_1.npz")),
    )


@pytest.fixture
def model_path(data_dir):
    return str(data_dir / "example_model_v2_regression_mol+mol.pt")


def test_train_quick(monkeypatch, data_path):
    input_path, desc_path, atom_feat_path_0, atom_feat_path_1, bond_feat_path_0, atom_desc_path_1 = data_path

    args = [
        "chemprop",
        "train",
        "-i",
        input_path,
        "--smiles-columns",
        "smiles",
        "solvent",
        "--epochs",
        "1",
        "--num-workers",
        "0",
        "--descriptors-path",
        desc_path,
        "--atom-features-path",
        *atom_feat_path_0,
        "--atom-features-path",
        *atom_feat_path_1,
        "--bond-features-path",
        *bond_feat_path_0,
        "--atom-descriptors-path",
        *atom_desc_path_1,
    ]

    with monkeypatch.context() as m:
        m.setattr("sys.argv", args)
        main()


def test_predict_quick(monkeypatch, data_path, model_path):
    input_path, _, _, _, _, _ = data_path

    args = [
        "chemprop",
        "predict",
        "-i",
        input_path,
        "--smiles-columns",
        "smiles",
        "solvent",
        "--model-path",
        model_path,
    ]

    with monkeypatch.context() as m:
        m.setattr("sys.argv", args)
        main()


def test_train_output_structure(monkeypatch, data_path, tmp_path):
    input_path, *_ = data_path

    args = [
        "chemprop",
        "train",
        "-i",
        input_path,
        "--smiles-columns",
        "smiles",
        "solvent",
        "--epochs",
        "1",
        "--num-workers",
        "0",
        "--save-dir",
        str(tmp_path),
        "--save-smiles-splits",
    ]

    with monkeypatch.context() as m:
        m.setattr("sys.argv", args)
        main()

    assert (tmp_path / "model_0" / "model.pt").exists()
    assert (tmp_path / "model_0" / "checkpoints" / "last.ckpt").exists()
    assert (tmp_path / "model_0" / "trainer_logs" / "version_0").exists()
    assert (tmp_path / "train_smiles.csv").exists()
