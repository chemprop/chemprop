"""This tests the CLI functionality of training and predicting a regression model on a single molecule.
"""

import pytest
import torch

from chemprop.cli.main import main
from chemprop.models.model import MPNN

pytestmark = pytest.mark.CLI


@pytest.fixture
def data_path(data_dir):
    return (
        str(data_dir / "regression" / "mol" / "mol.csv"),
        str(data_dir / "regression" / "mol" / "descriptors.npz"),
        str(data_dir / "regression" / "mol" / "atom_features.npz"),
        str(data_dir / "regression" / "mol" / "bond_features.npz"),
        str(data_dir / "regression" / "mol" / "atom_descriptors.npz"),
    )


@pytest.fixture
def model_path(data_dir):
    return str(data_dir / "example_model_v2_regression_mol.pt")


def test_train_quick(monkeypatch, data_path):
    input_path, *_ = data_path

    args = ["chemprop", "train", "-i", input_path, "--epochs", "1", "--num-workers", "0"]

    with monkeypatch.context() as m:
        m.setattr("sys.argv", args)
        main()


def test_train_quick_features(monkeypatch, data_path):
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
        "--epochs",
        "1",
        "--num-workers",
        "0",
        "--descriptors-path",
        descriptors_path,
        "--atom-features-path",
        atom_features_path,
        "--bond-features-path",
        bond_features_path,
        "--atom-descriptors-path",
        atom_descriptors_path,
    ]

    with monkeypatch.context() as m:
        m.setattr("sys.argv", args)
        main()


def test_predict_quick(monkeypatch, data_path, model_path):
    input_path, *_ = data_path
    args = ["chemprop", "predict", "-i", input_path, "--model-path", model_path]

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
        "--epochs",
        "1",
        "--num-workers",
        "0",
        "--save-dir",
        str(tmp_path),
        "--save-smiles-splits",
        "--save-preds",
    ]

    with monkeypatch.context() as m:
        m.setattr("sys.argv", args)
        main()

    assert (tmp_path / "model_0" / "model.pt").exists()
    assert (tmp_path / "model_0" / "checkpoints" / "last.ckpt").exists()
    assert (tmp_path / "model_0" / "trainer_logs" / "version_0").exists()
    assert (tmp_path / "train_smiles.csv").exists()
    assert (tmp_path / "model_0" / "test_predictions.csv").exists()


def test_train_output_structure_cv_ensemble(monkeypatch, data_path, tmp_path):
    input_path, *_ = data_path
    args = [
        "chemprop",
        "train",
        "-i",
        input_path,
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
        "--metrics",
        "mse",
        "rmse",
    ]

    with monkeypatch.context() as m:
        m.setattr("sys.argv", args)
        main()

    assert (tmp_path / "fold_2" / "model_1" / "model.pt").exists()
    assert (tmp_path / "fold_2" / "model_1" / "checkpoints" / "last.ckpt").exists()
    assert (tmp_path / "fold_2" / "model_1" / "trainer_logs" / "version_0").exists()
    assert (tmp_path / "fold_2" / "train_smiles.csv").exists()


def test_predict_output_structure(monkeypatch, data_path, model_path, tmp_path):
    input_path, *_ = data_path
    args = [
        "chemprop",
        "predict",
        "-i",
        input_path,
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
    input_path, *_ = data_path

    args = [
        "chemprop",
        "train",
        "-i",
        input_path,
        "--epochs",
        "1",
        "--num-workers",
        "0",
        "--save-dir",
        str(tmp_path),
    ]

    with monkeypatch.context() as m:
        m.setattr("sys.argv", args)
        main()

    checkpoint_path = tmp_path / "model_0" / "checkpoints" / "last.ckpt"

    model = MPNN.load_from_checkpoint(checkpoint_path)


def test_freeze_model(monkeypatch, data_path, model_path, tmp_path):
    input_path, *_ = data_path
    args = [
        "chemprop",
        "train",
        "-i",
        input_path,
        "--epochs",
        "1",
        "--num-workers",
        "0",
        "--save-dir",
        str(tmp_path),
        "--model-frzn",
        model_path,
        "--frzn-ffn-layers",
        "1",
    ]

    with monkeypatch.context() as m:
        m.setattr("sys.argv", args)
        main()

    checkpoint_path = tmp_path / "model_0" / "checkpoints" / "last.ckpt"

    trained_model = MPNN.load_from_checkpoint(checkpoint_path)
    frzn_model = MPNN.load_from_file(model_path)
    
    assert torch.equal(
        trained_model.message_passing.W_o.weight, frzn_model.message_passing.W_o.weight
    )
    assert torch.equal(trained_model.predictor.ffn[0].weight, frzn_model.predictor.ffn[0].weight)
