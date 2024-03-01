"""This tests the CLI functionality of training and predicting a regression model on a single molecule.
"""

import pytest

from chemprop.cli.main import main

pytestmark = pytest.mark.CLI


@pytest.fixture
def data_path(data_dir):
    return str(data_dir / "regression" / "mol_multitask.csv")


@pytest.fixture
def model_path(data_dir):
    return str(data_dir / "example_model_v2_regression_mol_multitask.pt")


def test_train_quick(monkeypatch, data_path):
    args = ["chemprop", "train", "-i", data_path, "--epochs", "1", "--num-workers", "0"]

    with monkeypatch.context() as m:
        m.setattr("sys.argv", args)
        main()


def test_predict_quick(monkeypatch, data_path, model_path):
    args = [
        "chemprop",
        "predict",
        "-i",
        data_path,
        "--model-path",
        model_path,
        "--target-columns",
        "mu",
        "alpha",
        "homo",
        "lumo",
        "gap",
        "r2",
        "zpve",
        "cv",
        "u0",
        "u298",
        "h298",
        "g298",
    ]

    with monkeypatch.context() as m:
        m.setattr("sys.argv", args)
        main()

    args = ["chemprop", "predict", "-i", data_path, "--model-path", model_path]

    with monkeypatch.context() as m:
        m.setattr("sys.argv", args)
        main()
