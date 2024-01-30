"""This tests the CLI functionality of training and predicting a regression model on a single molecule.
"""

import pytest

from chemprop.cli.main import main

pytestmark = pytest.mark.CLI


@pytest.fixture
def data_path(data_dir):
    return str(data_dir / "regression" / "mol.csv")


@pytest.fixture
def checkpoint_path(data_dir):
    return str(data_dir / "example_model_v2.ckpt")


def test_quick_train(monkeypatch, data_path):
    args = ["chemprop", "train", "-i", data_path, "--epochs", "1", "--num-workers", "4"]

    with monkeypatch.context() as m:
        m.setattr("sys.argv", args)
        main()


def test_quick_predict(monkeypatch, data_path, checkpoint_path):
    args = ["chemprop", "predict", "-i", data_path, "--checkpoint", checkpoint_path]

    with monkeypatch.context() as m:
        m.setattr("sys.argv", args)
        main()
