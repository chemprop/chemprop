"""This tests the CLI functionality of training and predicting a regression model on a single molecule.
"""

import pytest

from chemprop.cli.main import main
from chemprop.models.model import MPNN

pytestmark = pytest.mark.CLI


@pytest.fixture
def data_path(data_dir):
    return str(data_dir / "classification" / "mol_multiclass.csv")


@pytest.fixture
def model_path(data_dir):
    return str(data_dir / "example_model_v2_classification_mol_multiclass.pt")


def test_train_quick(monkeypatch, data_path):
    args = [
        "chemprop",
        "train",
        "-i",
        data_path,
        "--epochs",
        "1",
        "--num-workers",
        "0",
        "--task-type",
        "multiclass",
    ]

    with monkeypatch.context() as m:
        m.setattr("sys.argv", args)
        main()


def test_predict_quick(monkeypatch, data_path, model_path):
    args = ["chemprop", "predict", "-i", data_path, "--model-path", model_path]

    with monkeypatch.context() as m:
        m.setattr("sys.argv", args)
        main()


def test_train_output_structure(monkeypatch, data_path, tmp_path):
    args = [
        "chemprop",
        "train",
        "-i",
        data_path,
        "--epochs",
        "1",
        "--num-workers",
        "0",
        "--save-dir",
        str(tmp_path),
        "--save-smiles-splits",
        "--task-type",
        "multiclass",
    ]

    with monkeypatch.context() as m:
        m.setattr("sys.argv", args)
        main()

    assert (tmp_path / "model_0" / "model.pt").exists()
    assert (tmp_path / "model_0" / "checkpoints" / "last.ckpt").exists()
    assert (tmp_path / "model_0" / "trainer_logs" / "version_0").exists()
    assert (tmp_path / "train_smiles.csv").exists()


def test_train_output_structure_cv_ensemble(monkeypatch, data_path, tmp_path):
    args = [
        "chemprop",
        "train",
        "-i",
        data_path,
        "--epochs",
        "1",
        "--num-workers",
        "0",
        "--save-dir",
        str(tmp_path),
        "--save-smiles-splits",
        "--split-type",
        "cv",
        "--num-folds",
        "3",
        "--ensemble-size",
        "2",
        "--task-type",
        "multiclass",
    ]

    with monkeypatch.context() as m:
        m.setattr("sys.argv", args)
        main()

    assert (tmp_path / "fold_2" / "model_1" / "model.pt").exists()
    assert (tmp_path / "fold_2" / "model_1" / "checkpoints" / "last.ckpt").exists()
    assert (tmp_path / "fold_2" / "model_1" / "trainer_logs" / "version_0").exists()
    assert (tmp_path / "fold_2" / "train_smiles.csv").exists()


def test_predict_output_structure(monkeypatch, data_path, model_path, tmp_path):
    args = [
        "chemprop",
        "predict",
        "-i",
        data_path,
        "--model-path",
        model_path,
        "--output",
        str(tmp_path / "preds.csv"),
    ]

    with monkeypatch.context() as m:
        m.setattr("sys.argv", args)
        main()

    assert (tmp_path / "preds_0.csv").exists()


def test_train_outputs(monkeypatch, data_path, tmp_path):
    args = [
        "chemprop",
        "train",
        "-i",
        data_path,
        "--epochs",
        "1",
        "--num-workers",
        "0",
        "--save-dir",
        str(tmp_path),
        "--task-type",
        "multiclass",
    ]

    with monkeypatch.context() as m:
        m.setattr("sys.argv", args)
        main()

    checkpoint_path = tmp_path / "model_0" / "checkpoints" / "last.ckpt"

    model = MPNN.load_from_checkpoint(checkpoint_path)
