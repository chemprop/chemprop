"""This tests the CLI functionality of training and predicting a regression model on a single molecule.
"""

import pytest

from chemprop.cli.main import main
from chemprop.models.model import MPNN

pytestmark = pytest.mark.CLI


@pytest.fixture
def data_path(data_dir):
    return str(data_dir / "classification" / "mol.csv")


@pytest.fixture
def model_path(data_dir):
    return str(data_dir / "example_model_v2_classification_mol.pt")


@pytest.fixture
def dirichlet_model_path(data_dir):
    return str(data_dir / "example_model_v2_classification_dirichlet_mol.pt")


def test_train_quick(monkeypatch, data_path):
    base_args = [
        "chemprop",
        "train",
        "-i",
        data_path,
        "--epochs",
        "3",
        "--num-workers",
        "0",
        "--task-type",
        "classification",
        "--metric",
        "prc",
        "accuracy",
        "f1",
        "roc",
        "--show-individual-scores",
    ]

    task_types = ["classification", "classification-dirichlet"]

    for task_type in task_types:
        args = base_args.copy()

        args += ["--task-type", task_type]

        with monkeypatch.context() as m:
            m.setattr("sys.argv", args)
            main()


def test_predict_quick(monkeypatch, data_path, model_path):
    args = ["chemprop", "predict", "-i", data_path, "--model-path", model_path]

    with monkeypatch.context() as m:
        m.setattr("sys.argv", args)
        main()


def test_predict_ignore_stereo(monkeypatch, data_path, model_path):
    args = ["chemprop", "predict", "-i", data_path, "--model-path", model_path, "--ignore-stereo"]

    with monkeypatch.context() as m:
        m.setattr("sys.argv", args)
        main()


def test_predict_dirichlet_quick(monkeypatch, data_path, dirichlet_model_path):
    args = [
        "chemprop",
        "predict",
        "-i",
        data_path,
        "--model-path",
        dirichlet_model_path,
        "--uncertainty-method",
        "classification-dirichlet",
    ]

    with monkeypatch.context() as m:
        m.setattr("sys.argv", args)
        main()


@pytest.mark.parametrize("calibration_method", ["platt", "isotonic"])
def test_predict_unc_quick(monkeypatch, data_path, model_path, calibration_method):
    args = [
        "chemprop",
        "predict",
        "-i",
        data_path,
        "--model-path",
        model_path,
        "--cal-path",
        data_path,
        "--uncertainty-method",
        "classification",
        "--calibration-method",
        calibration_method,
        "--evaluation-methods",
        "nll-classification",
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
        "--model-path",
        model_path,
        "--ffn-block-index",
        ffn_block_index,
    ]

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
        "3",
        "--num-workers",
        "0",
        "--save-dir",
        str(tmp_path),
        "--save-smiles-splits",
        "--task-type",
        "classification",
    ]

    with monkeypatch.context() as m:
        m.setattr("sys.argv", args)
        main()

    assert (tmp_path / "model_0" / "best.pt").exists()
    assert (tmp_path / "model_0" / "checkpoints" / "last.ckpt").exists()
    assert (tmp_path / "model_0" / "trainer_logs" / "version_0").exists()
    assert (tmp_path / "train_smiles.csv").exists()


def test_train_output_structure_replicate_ensemble(monkeypatch, data_path, tmp_path):
    args = [
        "chemprop",
        "train",
        "-i",
        data_path,
        "--epochs",
        "3",
        "--num-workers",
        "0",
        "--save-dir",
        str(tmp_path),
        "--save-smiles-splits",
        "--split-type",
        "random",
        "--num-replicates",
        "3",
        "--ensemble-size",
        "2",
        "--task-type",
        "classification",
    ]

    with monkeypatch.context() as m:
        m.setattr("sys.argv", args)
        main()

    assert (tmp_path / "replicate_2" / "model_1" / "best.pt").exists()
    assert (tmp_path / "replicate_2" / "model_1" / "checkpoints" / "last.ckpt").exists()
    assert (tmp_path / "replicate_2" / "model_1" / "trainer_logs" / "version_0").exists()
    assert (tmp_path / "replicate_2" / "train_smiles.csv").exists()


def test_predict_output_structure(monkeypatch, data_path, model_path, tmp_path):
    args = [
        "chemprop",
        "predict",
        "-i",
        data_path,
        "--model-path",
        model_path,
        model_path,
        "--output",
        str(tmp_path / "preds.csv"),
    ]

    with monkeypatch.context() as m:
        m.setattr("sys.argv", args)
        main()

    assert (tmp_path / "preds.csv").exists()
    assert (tmp_path / "preds_individual.csv").exists()


@pytest.mark.parametrize("ffn_block_index", ["0", "1"])
def test_fingerprint_output_structure(
    monkeypatch, data_path, model_path, tmp_path, ffn_block_index
):
    args = [
        "chemprop",
        "fingerprint",
        "-i",
        data_path,
        "--model-path",
        model_path,
        "--output",
        str(tmp_path / "fps.csv"),
        "--ffn-block-index",
        ffn_block_index,
    ]

    with monkeypatch.context() as m:
        m.setattr("sys.argv", args)
        main()

    assert (tmp_path / "fps_0.csv").exists()


def test_train_outputs(monkeypatch, data_path, tmp_path):
    args = [
        "chemprop",
        "train",
        "-i",
        data_path,
        "--epochs",
        "3",
        "--num-workers",
        "0",
        "--save-dir",
        str(tmp_path),
        "--task-type",
        "classification",
    ]

    with monkeypatch.context() as m:
        m.setattr("sys.argv", args)
        main()

    checkpoint_path = tmp_path / "model_0" / "checkpoints" / "last.ckpt"

    model = MPNN.load_from_checkpoint(checkpoint_path)
    assert model is not None


def test_class_balance(monkeypatch, data_path, tmp_path):
    args = [
        "chemprop",
        "train",
        "-i",
        data_path,
        "--epochs",
        "3",
        "--num-workers",
        "0",
        "--save-dir",
        str(tmp_path),
        "--task-type",
        "classification",
        "--class-balance",
    ]

    with monkeypatch.context() as m:
        m.setattr("sys.argv", args)
        main()
