"""This tests the CLI functionality of training and predicting a regression model on a single polymer.
"""

import pytest

from chemprop.cli.hpopt import NO_HYPEROPT, NO_OPTUNA, NO_RAY
from chemprop.cli.main import main
from chemprop.cli.train import TrainSubcommand
from chemprop.models.model import MPNN

pytestmark = pytest.mark.CLI


@pytest.fixture
def data_path(data_dir):
    return (
        str(data_dir / "regression" / "polymer" / "dataset-poly_chemprop_small.csv"),
    )

@pytest.fixture
def model_path(data_dir):
    return str(data_dir / "regression" / "polymer" / "example_model_v2_regression_polymer.ckpt")


@pytest.fixture
def config_path(data_dir):
    return str(data_dir / "regression" / "polymer" / "config.toml")


def test_train_quick(monkeypatch, data_path):
    input_path, *_ = data_path

    args = [
        "chemprop",
        "train",
        "-i",
        input_path,
        "--epochs",
        "3",
        "--num-workers",
        "0",
        "--show-individual-scores",
        "--polymer-columns",
        "poly_chemprop_input",
    ]

    with monkeypatch.context() as m:
        m.setattr("sys.argv", args)
        main()


def test_train_config(monkeypatch, config_path, tmp_path):
    args = [
        "chemprop",
        "train",
        "--config-path",
        config_path,
        "--epochs",
        "3",
        "--num-workers",
        "0",
        "--save-dir",
        str(tmp_path),
        "--polymer-columns",
        "poly_chemprop_input",
    ]

    with monkeypatch.context() as m:
        m.setattr("sys.argv", args)
        main()

    new_config_path = tmp_path / "config.toml"
    parser = TrainSubcommand.parser

    new_args = parser.parse_args(["--config-path", str(new_config_path)])
    old_args = parser.parse_args(["--config-path", str(config_path)])

    for key, value in old_args.__dict__.items():
        if key not in ["config_path", "output_dir", "epochs"]:
            assert getattr(new_args, key) == value

    assert new_args.epochs == 3


def test_predict_quick(monkeypatch, data_path, model_path):
    input_path, *_ = data_path
    args = ["chemprop", 
            "predict",
            "-i",
            input_path,
            "--model-path",
            model_path,
            "--polymer-columns",
            "poly_chemprop_input",
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
        "--model-path",
        model_path,
        "--ffn-block-index",
        ffn_block_index,
        "--polymer-columns",
        "poly_chemprop_input",
    ]

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
        "3",
        "--num-workers",
        "0",
        "--save-dir",
        str(tmp_path),
        "--save-smiles-splits",
        "--polymer-columns",
        "poly_chemprop_input",
    ]

    with monkeypatch.context() as m:
        m.setattr("sys.argv", args)
        main()

    assert (tmp_path / "model_0" / "best.pt").exists()
    assert (tmp_path / "model_0" / "checkpoints" / "last.ckpt").exists()
    assert (tmp_path / "model_0" / "trainer_logs" / "version_0").exists()
    assert (tmp_path / "train_smiles.csv").exists()
    assert (tmp_path / "model_0" / "test_predictions.csv").exists()


def test_train_output_structure_replicate_ensemble(monkeypatch, data_path, tmp_path):
    input_path, *_ = data_path
    args = [
        "chemprop",
        "train",
        "-i",
        input_path,
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
        "--metrics",
        "mse",
        "rmse",
        "--molecule-featurizers",
        "rdkit_2d",
        "--polymer-columns",
        "poly_chemprop_input",
    ]

    with monkeypatch.context() as m:
        m.setattr("sys.argv", args)
        main()

    assert (tmp_path / "replicate_2" / "model_1" / "best.pt").exists()
    assert (tmp_path / "replicate_2" / "model_1" / "checkpoints" / "last.ckpt").exists()
    assert (tmp_path / "replicate_2" / "model_1" / "trainer_logs" / "version_0").exists()
    assert (tmp_path / "replicate_2" / "train_smiles.csv").exists()



def test_predict_output_structure(monkeypatch, data_path, model_path, tmp_path):
    input_path, *_ = data_path
    args = [
        "chemprop",
        "predict",
        "-i",
        input_path,
        "--model-path",
        model_path,
        model_path, 
        "--polymer-columns",
        "poly_chemprop_input",
        "--output",
        str(tmp_path / "preds.csv"),
    ]

    with monkeypatch.context() as m:
        m.setattr("sys.argv", args)
        main()

    assert (tmp_path / "preds.csv").exists()
    assert (tmp_path / "preds_individual.csv").exists()


def test_train_outputs(monkeypatch, data_path, tmp_path):
    input_path, *_ = data_path

    args = [
        "chemprop",
        "train",
        "-i",
        input_path,
        "--epochs",
        "3",
        "--num-workers",
        "0",
        "--save-dir",
        str(tmp_path),
        "--polymer-columns",
        "poly_chemprop_input",
    ]

    with monkeypatch.context() as m:
        m.setattr("sys.argv", args)
        main()

    checkpoint_path = tmp_path / "model_0" / "checkpoints" / "last.ckpt"

    model = MPNN.load_from_checkpoint(checkpoint_path)
    assert model is not None



def test_checkpoint_model(monkeypatch, data_path, model_path, tmp_path):
    input_path, *_ = data_path
    args = [
        "chemprop",
        "train",
        "-i",
        input_path,
        "--epochs",
        "3",
        "--num-workers",
        "0",
        "--save-dir",
        str(tmp_path),
        "--checkpoint",
        model_path,
        "--polymer-columns",
        "poly_chemprop_input",
        "--target-columns",
        "EA vs SHE (eV)"
    ]

    with monkeypatch.context() as m:
        m.setattr("sys.argv", args)
        main()

    checkpoint_path = tmp_path / "model_0" / "checkpoints" / "last.ckpt"

    model = MPNN.load_from_checkpoint(checkpoint_path)
    assert model is not None


@pytest.mark.skipif(NO_RAY or NO_OPTUNA, reason="Optuna not installed")
def test_optuna_quick(monkeypatch, data_path, tmp_path):
    input_path, *_ = data_path

    args = [
        "chemprop",
        "hpopt",
        "-i",
        input_path,
        "--epochs",
        "6",
        "--hpopt-save-dir",
        str(tmp_path),
        "--raytune-num-samples",
        "2",
        "--raytune-search-algorithm",
        "optuna",
        "--search-parameter-keywords",
        "all",
        "--polymer-columns",
        "poly_chemprop_input",
    ]

    with monkeypatch.context() as m:
        m.setattr("sys.argv", args)
        main()

    assert (tmp_path / "best_config.toml").exists()
    assert (tmp_path / "best_checkpoint.ckpt").exists()
    assert (tmp_path / "all_progress.csv").exists()
    assert (tmp_path / "ray_results").exists()

    args = [
        "chemprop",
        "train",
        "--config-path",
        str(tmp_path / "best_config.toml"),
        "--save-dir",
        str(tmp_path),
        "--polymer-columns",
        "poly_chemprop_input",
    ]

    with monkeypatch.context() as m:
        m.setattr("sys.argv", args)
        main()

    assert (tmp_path / "model_0" / "best.pt").exists()


@pytest.mark.skipif(NO_RAY or NO_HYPEROPT, reason="Ray and/or Hyperopt not installed")
def test_hyperopt_quick(monkeypatch, data_path, tmp_path):
    input_path, *_ = data_path

    args = [
        "chemprop",
        "hpopt",
        "-i",
        input_path,
        "--epochs",
        "6",
        "--hpopt-save-dir",
        str(tmp_path),
        "--raytune-num-samples",
        "2",
        "--raytune-search-algorithm",
        "hyperopt",
        "--molecule-featurizers",
        "morgan_binary",
        "--search-parameter-keywords",
        "all",
        "--polymer-columns",
        "poly_chemprop_input",
    ]

    with monkeypatch.context() as m:
        m.setattr("sys.argv", args)
        main()

    assert (tmp_path / "best_config.toml").exists()
    assert (tmp_path / "best_checkpoint.ckpt").exists()
    assert (tmp_path / "all_progress.csv").exists()
    assert (tmp_path / "ray_results").exists()

    args = [
        "chemprop",
        "train",
        "--config-path",
        str(tmp_path / "best_config.toml"),
        "--save-dir",
        str(tmp_path),
        "--polymer-columns",
        "poly_chemprop_input",
    ]

    with monkeypatch.context() as m:
        m.setattr("sys.argv", args)
        main()

    assert (tmp_path / "model_0" / "best.pt").exists()
