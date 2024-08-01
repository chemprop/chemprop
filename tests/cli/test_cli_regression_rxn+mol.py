"""This tests the CLI functionality of training and predicting a regression model on a multi-molecule.
"""

import pytest

from chemprop.cli.main import main

pytestmark = pytest.mark.CLI


@pytest.fixture
def data_path(data_dir):
    return (
        str(data_dir / "regression" / "rxn+mol" / "rxn+mol.csv"),
        str(data_dir / "regression" / "rxn+mol" / "descriptors.npz"),
        ("0", str(data_dir / "regression" / "rxn+mol" / "atom_features.npz")),
        ("0", str(data_dir / "regression" / "rxn+mol" / "bond_features.npz")),
        ("0", str(data_dir / "regression" / "rxn+mol" / "atom_descriptors.npz")),
    )


@pytest.fixture
def model_path(data_dir):
    return str(data_dir / "example_model_v2_regression_rxn+mol.pt")


def test_train_quick(monkeypatch, data_path):
    (
        input_path,
        descriptors_path,
        atom_features_path,
        bond_features_path,
        atom_descriptors_path,
    ) = data_path
    args = [
        "chemprop",
        "train",
        "-i",
        input_path,
        "--reaction-columns",
        "rxn_smiles",
        "--smiles-columns",
        "solvent_smiles",
        "--epochs",
        "1",
        "--num-workers",
        "0",
        "--split-key-molecule",
        "1",
        "--descriptors-path",
        descriptors_path,
        "--atom-features-path",
        *atom_features_path,
        "--bond-features-path",
        *bond_features_path,
        "--atom-descriptors-path",
        *atom_descriptors_path,
        "--show-individual-scores",
    ]

    with monkeypatch.context() as m:
        m.setattr("sys.argv", args)
        main()


def test_predict_quick(monkeypatch, data_path, model_path):
    input_path, *_ = data_path
    args = [
        "chemprop",
        "predict",
        "-i",
        input_path,
        "--reaction-columns",
        "rxn_smiles",
        "--smiles-columns",
        "solvent_smiles",
        "--model-path",
        model_path,
    ]

    with monkeypatch.context() as m:
        m.setattr("sys.argv", args)
        main()


@pytest.mark.parametrize("ffn_block_index", ["0", "1"])
def test_fingerprint_quick(monkeypatch, data_path, model_path, ffn_block_index):
    input_path, *_ = data_path
    args = [
        "chemprop",
        "fingerprint",
        "-i",
        input_path,
        "--reaction-columns",
        "rxn_smiles",
        "--smiles-columns",
        "solvent_smiles",
        "--model-path",
        model_path,
        "--ffn-block-index",
        ffn_block_index,
    ]

    with monkeypatch.context() as m:
        m.setattr("sys.argv", args)
        main()


def test_train_molecule_featurizers(monkeypatch, data_path):
    input_path, descriptors_path, *_ = data_path
    args = [
        "chemprop",
        "train",
        "-i",
        input_path,
        "--reaction-columns",
        "rxn_smiles",
        "--smiles-columns",
        "solvent_smiles",
        "--epochs",
        "1",
        "--num-workers",
        "0",
        "--split-key-molecule",
        "1",
        "--descriptors-path",
        descriptors_path,
        "--molecule-featurizers",
        "morgan_count",
    ]

    with monkeypatch.context() as m:
        m.setattr("sys.argv", args)
        main()
