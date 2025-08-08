"""This tests the CLI functionality of training and predicting a regression model on a single molecule.
"""

import json

import pytest
import torch

from chemprop.cli.hpopt import NO_HYPEROPT, NO_OPTUNA, NO_RAY
from chemprop.cli.main import main
from chemprop.cli.train import FoundationModels, TrainSubcommand
from chemprop.models.model import MPNN
from chemprop.utils.utils import is_cuikmolmaker_available

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


@pytest.fixture
def extra_model_path(data_dir):
    return str(data_dir / "example_model_v2_regression_mol_with_metrics.ckpt")


@pytest.fixture
def mve_model_path(data_dir):
    return str(data_dir / "example_model_v2_regression_mve_mol.pt")


@pytest.fixture
def evidential_model_path(data_dir):
    return str(data_dir / "example_model_v2_regression_evidential_mol.pt")


@pytest.fixture
def quantile_model_path(data_dir):
    return str(data_dir / "example_model_v2_regression_quantile_mol.pt")


@pytest.fixture
def config_path(data_dir):
    return str(data_dir / "regression" / "mol" / "config.toml")


@pytest.fixture
def data_with_descriptors_path(data_dir):
    return str(data_dir / "regression" / "mol" / "mol_with_descriptors.csv")


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
    ]

    with monkeypatch.context() as m:
        m.setattr("sys.argv", args)
        main()


def test_train_quick_from_foundation(monkeypatch, data_path):
    input_path, *_ = data_path

    for foundation in FoundationModels.keys():
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
            "--from-foundation",
            foundation,
        ]

        with monkeypatch.context() as m:
            m.setattr("sys.argv", args)
            main()


def test_train_quick_from_local_foundation(monkeypatch, data_path, model_path):
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
        "--from-foundation",
        model_path,
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


def test_train_quick_features(monkeypatch, data_path):
    (
        input_path,
        descriptors_path,
        atom_features_path,
        bond_features_path,
        atom_descriptors_path,
    ) = data_path

    base_args = [
        "chemprop",
        "train",
        "-i",
        input_path,
        "--epochs",
        "3",
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

    task_types = ["", "regression-mve", "regression-evidential", "regression-quantile"]

    for task_type in task_types:
        args = base_args.copy()

        if task_type:
            args += ["--task-type", task_type]

        if task_type == "regression-evidential":
            args += ["--evidential-regularization", "0.2"]

        with monkeypatch.context() as m:
            m.setattr("sys.argv", args)
            main()


@pytest.mark.skipif(not is_cuikmolmaker_available(), reason="cuik_molmaker not installed")
def test_train_quick_features_cuikmolmaker(monkeypatch, data_path):
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
        "3",
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
        "--use-cuikmolmaker-featurization",
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


@pytest.mark.skipif(not is_cuikmolmaker_available(), reason="cuik_molmaker not installed")
def test_predict_quick_cuikmolmaker(monkeypatch, data_path, model_path):
    input_path, *_ = data_path
    args = [
        "chemprop",
        "predict",
        "-i",
        input_path,
        "--model-path",
        model_path,
        "--use-cuikmolmaker-featurization",
    ]

    with monkeypatch.context() as m:
        m.setattr("sys.argv", args)
        main()


def test_predict_ensemble_quick(monkeypatch, data_path, model_path, extra_model_path):
    input_path, *_ = data_path
    args = [
        "chemprop",
        "predict",
        "-i",
        input_path,
        "--model-path",
        model_path,
        extra_model_path,
        "--uncertainty-method",
        "ensemble",
    ]

    with monkeypatch.context() as m:
        m.setattr("sys.argv", args)
        main()


def test_predict_dropout_quick(monkeypatch, data_path, model_path, extra_model_path):
    input_path, *_ = data_path
    args = [
        "chemprop",
        "predict",
        "-i",
        input_path,
        "--model-path",
        model_path,
        extra_model_path,
        "--uncertainty-method",
        "dropout",
    ]

    with monkeypatch.context() as m:
        m.setattr("sys.argv", args)
        main()


def test_predict_mve_quick(monkeypatch, data_path, mve_model_path):
    input_path, *_ = data_path
    args = [
        "chemprop",
        "predict",
        "-i",
        input_path,
        "--model-path",
        mve_model_path,
        "--cal-path",
        input_path,
        "--uncertainty-method",
        "mve",
        "--calibration-method",
        "zscaling",
        "--evaluation-methods",
        "nll-regression",
        "miscalibration_area",
        "ence",
        "spearman",
    ]

    with monkeypatch.context() as m:
        m.setattr("sys.argv", args)
        main()


def test_predict_evidential_quick(monkeypatch, data_path, evidential_model_path):
    input_path, *_ = data_path
    args = [
        "chemprop",
        "predict",
        "-i",
        input_path,
        "--model-path",
        evidential_model_path,
        "--cal-path",
        input_path,
        "--uncertainty-method",
        "evidential-total",
        "--calibration-method",
        "zscaling",
        "--evaluation-methods",
        "nll-regression",
        "miscalibration_area",
        "ence",
        "spearman",
    ]

    with monkeypatch.context() as m:
        m.setattr("sys.argv", args)
        main()


def test_predict_quantile_quick(monkeypatch, data_path, quantile_model_path):
    input_path, *_ = data_path
    args = [
        "chemprop",
        "predict",
        "-i",
        input_path,
        "--model-path",
        quantile_model_path,
        "--cal-path",
        input_path,
        "--uncertainty-method",
        "quantile-regression",
        "--calibration-method",
        "conformal-regression",
        "--conformal-alpha",
        "0.1",
        "--evaluation-methods",
        "spearman",
        "conformal-coverage-regression",
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
    ]

    with monkeypatch.context() as m:
        m.setattr("sys.argv", args)
        main()

    assert (tmp_path / "replicate_2" / "model_1" / "best.pt").exists()
    assert (tmp_path / "replicate_2" / "model_1" / "checkpoints" / "last.ckpt").exists()
    assert (tmp_path / "replicate_2" / "model_1" / "trainer_logs" / "version_0").exists()
    assert (tmp_path / "replicate_2" / "train_smiles.csv").exists()


def test_train_csv_splits(monkeypatch, data_dir, tmp_path):
    input_path = str(data_dir / "regression" / "mol" / "mol_with_splits.csv")
    args = [
        "chemprop",
        "train",
        "-i",
        input_path,
        "--smiles-columns",
        "smiles",
        "--target-columns",
        "lipo",
        "--splits-column",
        "split",
        "--epochs",
        "3",
        "--num-workers",
        "0",
        "--save-dir",
        str(tmp_path),
    ]

    with monkeypatch.context() as m:
        m.setattr("sys.argv", args)
        main()


def test_train_splits_file(monkeypatch, data_path, tmp_path):
    input_path, *_ = data_path
    splits_file = str(tmp_path / "splits.json")
    splits = [
        {"train": [0, 1], "val": "2-3", "test": "4,5"},
        {"val": [0, 1], "test": "2-3", "train": "4,5"},
    ]

    with open(splits_file, "w") as f:
        json.dump(splits, f)

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
        "--splits-file",
        splits_file,
    ]

    with monkeypatch.context() as m:
        m.setattr("sys.argv", args)
        main()


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
    input_path, *_ = data_path
    args = [
        "chemprop",
        "fingerprint",
        "-i",
        input_path,
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
    ]

    with monkeypatch.context() as m:
        m.setattr("sys.argv", args)
        main()

    checkpoint_path = tmp_path / "model_0" / "checkpoints" / "last.ckpt"

    model = MPNN.load_from_checkpoint(checkpoint_path)
    assert model is not None


def test_freeze_model(monkeypatch, data_path, model_path, tmp_path):
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
        "--freeze-encoder",
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
    assert torch.equal(
        trained_model.predictor.ffn[0][0].weight, frzn_model.predictor.ffn[0][0].weight
    )


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
        "--molecule-featurizers",
        "morgan_count",
        "--search-parameter-keywords",
        "all",
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
    ]

    with monkeypatch.context() as m:
        m.setattr("sys.argv", args)
        main()

    assert (tmp_path / "model_0" / "best.pt").exists()


@pytest.mark.skipif(NO_RAY or NO_HYPEROPT, reason="Ray and/or Hyperopt not installed")
def test_hyperopt_quick(monkeypatch, data_path, tmp_path):
    (
        input_path,
        descriptors_path,
        atom_features_path,
        bond_features_path,
        atom_descriptors_path,
    ) = data_path

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
    ]

    with monkeypatch.context() as m:
        m.setattr("sys.argv", args)
        main()

    assert (tmp_path / "model_0" / "best.pt").exists()


def test_custom_activation_quick(monkeypatch, data_path):
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
        "--activation",
        "SOFTPLUS",
        "--activation-args",
        "1.0",
        "threshold=15",
    ]

    with monkeypatch.context() as m:
        m.setattr("sys.argv", args)
        main()


def test_empty_testset(monkeypatch, data_path):
    input_path, *_ = data_path
    args = [
        "chemprop",
        "train",
        "-i",
        input_path,
        "--smiles-columns",
        "smiles",
        "--target-columns",
        "lipo",
        "--split-sizes",
        "0.5",
        "0.5",
        "0",
    ]

    with monkeypatch.context() as m:
        m.setattr("sys.argv", args)
        main()


def test_descriptors_columns(monkeypatch, data_with_descriptors_path):
    args = [
        "chemprop",
        "train",
        "-i",
        data_with_descriptors_path,
        "--target-columns",
        "y",
        "--descriptors-columns",
        "temperature",
        "pressure",
        "--splits-column",
        "split",
        "--epochs",
        "3",
    ]

    with monkeypatch.context() as m:
        m.setattr("sys.argv", args)
        main()


def test_descriptors_multisource(monkeypatch, data_path):
    input_path, descriptors_path, *_ = data_path
    args = [
        "chemprop",
        "train",
        "-i",
        input_path,
        "--target-columns",
        "lipo",
        "--descriptors-columns",
        "lipo",
        "--descriptors-path",
        descriptors_path,
        "--epochs",
        "3",
    ]

    with monkeypatch.context() as m:
        m.setattr("sys.argv", args)
        main()
