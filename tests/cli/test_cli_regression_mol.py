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
def checkpoint_path(data_dir):
    return str(data_dir / "example_model_v2.ckpt")


def test_train_quick(monkeypatch, data_path):
    args = ["chemprop", "train", "-i", data_path, "--epochs", "1", "--num-workers", "0"]

    with monkeypatch.context() as m:
        m.setattr("sys.argv", args)
        main()


def test_predict_quick(monkeypatch, data_path, checkpoint_path):
    args = ["chemprop", "predict", "-i", data_path, "--checkpoint", checkpoint_path]

    with monkeypatch.context() as m:
        m.setattr("sys.argv", args)
        main()


def test_train_output_structure(monkeypatch, data_path, tmp_path):
    args = ["chemprop", "train", "-i", data_path, "--epochs", "1", "--num-workers", "0", "--save-dir", str(tmp_path)]

    with monkeypatch.context() as m:
        m.setattr("sys.argv", args)
        main()

    assert (tmp_path / "logs" / "chemprop").exists()
    assert (tmp_path / "mol" / "model.pt").exists()
    assert (tmp_path / "mol" / "chkpts" / "last.ckpt").exists()
    assert (tmp_path / "mol" / "tb_logs" / "version_0").exists()


def test_predict_output_structure(monkeypatch, data_path, tmp_path):
    args = ["chemprop", "predict", "-i", data_path, "--checkpoint", "tests/data/example_model_v2.ckpt", "--output", str(tmp_path / "preds.csv")]

    with monkeypatch.context() as m:
        m.setattr("sys.argv", args)
        main()

    assert (tmp_path / "logs" / "chemprop").exists()
    assert (tmp_path / "lightning_logs").exists()
    assert (tmp_path / "preds.csv").exists()


def test_train_outputs(monkeypatch, data_path, tmp_path):
    args = ["chemprop", "train", "-i", data_path, "--epochs", "1", "--num-workers", "0", "--save-dir", str(tmp_path)]

    with monkeypatch.context() as m:
        m.setattr("sys.argv", args)
        main()

    checkpoint_path = tmp_path / "mol" / "chkpts" / "last.ckpt"

    model = MPNN.load_from_checkpoint(checkpoint_path)
