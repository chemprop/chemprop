"""This tests the CLI functionality of training and predicting a regression model on a multi-molecule.
"""

import pytest

from chemprop.cli.main import main

pytestmark = pytest.mark.CLI


@pytest.fixture
def data_path(data_dir):
    return str(data_dir / "regression" / "rxn+mol.csv")


@pytest.fixture
def model_path(data_dir):
    return str(data_dir / "example_model_v2_regression_rxn+mol.pt")


def test_train_quick(monkeypatch, data_path):
    args = [
        "chemprop",
        "train",
        "-i",
        data_path,
        "--reaction-columns",
        "rxn_smiles",
        "--smiles-columns",
        "solvent_smiles",
        "--epochs",
        "1",
        "--num-workers",
        "0",
        "--save-preds",
    ]

    with monkeypatch.context() as m:
        m.setattr("sys.argv", args)
        main()


def test_predict_quick(monkeypatch, data_path, model_path):
    args = [
        "chemprop",
        "predict",
        "-i",
        data_path,
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
    args = [
        "chemprop",
        "fingerprint",
        "-i",
        data_path,
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
