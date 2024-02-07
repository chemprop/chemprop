"""This tests the CLI functionality of training and predicting a regression model on a single molecule.
"""

import pytest

from chemprop.cli.main import main
from chemprop.models.model import MPNN

pytestmark = pytest.mark.CLI


@pytest.fixture
def data_path(data_dir):
    return str(data_dir / "regression" / "mol.csv")


@pytest.fixture
def model_path(data_dir):
    return str(data_dir / "example_model_v2_regression_mol.pt")


def test_train_quick(monkeypatch, data_path):
    args = ["chemprop", "train", "-i", data_path, "--epochs", "1", "--num-workers", "0"]

    with monkeypatch.context() as m:
        m.setattr("sys.argv", args)
        main()


def test_predict_quick(monkeypatch, data_path, model_path):
    args = ["chemprop", "predict", "-i", data_path, "--model-path", model_path]

    with monkeypatch.context() as m:
        m.setattr("sys.argv", args)
        main()


def test_train_output_structure(monkeypatch, data_path, tmp_path):
    args = ["chemprop", "train", "-i", data_path, "--epochs", "1", "--num-workers", "0", "--save-dir", str(tmp_path), "--save-smiles-splits"]

    with monkeypatch.context() as m:
        m.setattr("sys.argv", args)
        main()

    assert (tmp_path / "model.pt").exists()
    assert (tmp_path / "chkpts" / "last.ckpt").exists()
    assert (tmp_path / "trainer_logs" / "version_0").exists()
    assert (tmp_path / "train_smiles.csv").exists()


def test_predict_output_structure(monkeypatch, data_path, model_path, tmp_path):
    args = ["chemprop", "predict", "-i", data_path, "--model-path", model_path, "--output", str(tmp_path / "preds.csv")]

    with monkeypatch.context() as m:
        m.setattr("sys.argv", args)
        main()

    assert (tmp_path / "preds.csv").exists()


def test_train_outputs(monkeypatch, data_path, tmp_path):
    args = ["chemprop", "train", "-i", data_path, "--epochs", "1", "--num-workers", "0", "--save-dir", str(tmp_path)]

    with monkeypatch.context() as m:
        m.setattr("sys.argv", args)
        main()

    checkpoint_path = tmp_path / "chkpts" / "last.ckpt"

    model = MPNN.load_from_checkpoint(checkpoint_path)
