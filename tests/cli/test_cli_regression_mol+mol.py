"""This tests the CLI functionality of training and predicting a regression model on a multi-molecule.
"""

import pytest

from chemprop.cli.main import main

pytestmark = pytest.mark.CLI


@pytest.fixture
def data_path(data_dir):
    return str(data_dir / "regression" / "mol+mol.csv")


@pytest.fixture
def checkpoint_path(data_dir):
    return str(data_dir / "example_model_v2_regression_multi.ckpt")


def test_train(monkeypatch, data_path):
    args = ["chemprop", "train", "-i", data_path, "--smiles-columns", "smiles", "solvent", "--epochs", "1", "--num-workers", "4"]

    with monkeypatch.context() as m:
        m.setattr("sys.argv", args)
        main()


def test_predicting(monkeypatch, data_path, checkpoint_path):
    args = ["chemprop", "predict", "-i", data_path, "--smiles-columns", "smiles", "solvent", "--checkpoint", checkpoint_path]

    with monkeypatch.context() as m:
        m.setattr("sys.argv", args)
        main()
