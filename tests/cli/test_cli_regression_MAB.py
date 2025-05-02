"""This tests the CLI functionality of training and predicting a regression model on a single molecule.
"""

import json

import pytest

from chemprop.cli.hpopt import NO_HYPEROPT, NO_OPTUNA, NO_RAY
from chemprop.cli.main import main
from chemprop.models import MolAtomBondMPNN

pytestmark = pytest.mark.CLI


@pytest.fixture
def regression_data_path(data_dir):
    return (
        str(data_dir / "mol_atom_bond" / "regression.csv"),
        "smiles",
        ["mol_y1", "mol_y2"],
        ["atom_y1", "atom_y2"],
        ["bond_y1", "bond_y2"],
        "weight",
    )


@pytest.fixture
def bounded_data_path(data_dir):
    return (
        str(data_dir / "mol_atom_bond" / "bounded.csv"),
        "smiles",
        ["mol_y1", "mol_y2"],
        ["atom_y1", "atom_y2"],
        ["bond_y1", "bond_y2"],
        "weight",
    )


@pytest.fixture
def classification_data_path(data_dir):
    return (
        str(data_dir / "mol_atom_bond" / "classification.csv"),
        "smiles",
        ["mol_y1", "mol_y2"],
        ["atom_y1", "atom_y2"],
        ["bond_y1", "bond_y2"],
        "weight",
    )


@pytest.fixture
def multiclass_data_path(data_dir):
    return (
        str(data_dir / "mol_atom_bond" / "multiclass.csv"),
        "smiles",
        ["mol_y1", "mol_y2"],
        ["atom_y1", "atom_y2"],
        ["bond_y1", "bond_y2"],
        "weight",
    )


@pytest.fixture
def constrained_data_path(data_dir):
    return (
        str(data_dir / "mol_atom_bond" / "constrained_regression.csv"),
        str(data_dir / "mol_atom_bond" / "constrained_regression_constraints.csv"),
        ["atom_target_0", "atom_target_1", "bond_target_1"],
        "smiles",
        ["mol_y1"],
        ["atom_y1", "atom_y2"],
        ["bond_y1", "bond_y2"],
    )


@pytest.fixture
def extras_paths(data_dir):
    return (
        str(data_dir / "mol_atom_bond" / "descriptors.npz"),
        str(data_dir / "mol_atom_bond" / "atom_features_descriptors.npz"),
        str(data_dir / "mol_atom_bond" / "bond_features_descriptors.npz"),
        str(data_dir / "mol_atom_bond" / "atom_features_descriptors.npz"),
        str(data_dir / "mol_atom_bond" / "bond_features_descriptors.npz"),
    )


@pytest.fixture
def model_dir(data_dir):
    return data_dir / "mol_atom_bond" / "example_models"


@pytest.fixture
def regression_model_path(model_dir):
    return str(model_dir / "regression.pt")


@pytest.fixture
def regression_no_mol_model_path(model_dir):
    return str(model_dir / "regression_no_mol.pt")


@pytest.fixture
def regression_no_atom_model_path(model_dir):
    return str(model_dir / "regression_no_atom.pt")


@pytest.fixture
def regression_no_bond_model_path(model_dir):
    return str(model_dir / "regression_no_bond.pt")


@pytest.fixture
def regression_only_atom_model_path(model_dir):
    return str(model_dir / "regression_only_atom.pt")


@pytest.fixture
def regression_only_bond_model_path(model_dir):
    return str(model_dir / "regression_only_bond.pt")


@pytest.fixture
def classification_model_model_path(model_dir):
    return str(model_dir / "classification.pt")


@pytest.fixture
def multiclass_model_path(model_dir):
    return str(model_dir / "multiclass.pt")


@pytest.fixture
def constrained_model_path(model_dir):
    return str(model_dir / "regression_constrained.pt")


@pytest.fixture
def mve_model_path(model_dir):
    return str(model_dir / "regression_mve.pt")


def test_train_regression_quick(monkeypatch, regression_data_path):
    input_path, smiles, targets, atom_targets, bond_targets, weight = regression_data_path

    args = [
        "chemprop",
        "train",
        "-i",
        input_path,
        "--smiles-columns",
        smiles,
        "--target-columns",
        *targets,
        "--atom-target-columns",
        *atom_targets,
        "--bond-target-columns",
        *bond_targets,
        "--weight-column",
        weight,
        "--epochs",
        "3",
        "--show-individual-scores",
    ]

    with monkeypatch.context() as m:
        m.setattr("sys.argv", args)
        main()


def test_train_regression_quick_features(monkeypatch, regression_data_path, extras_paths):
    input_path, smiles, targets, atom_targets, bond_targets, weight = regression_data_path
    (
        descriptors_path,
        atom_features_path,
        bond_features_path,
        atom_descriptors_path,
        bond_descriptors_path,
    ) = extras_paths

    args = [
        "chemprop",
        "train",
        "-i",
        input_path,
        "--smiles-columns",
        smiles,
        "--target-columns",
        *targets,
        "--atom-target-columns",
        *atom_targets,
        "--bond-target-columns",
        *bond_targets,
        "--epochs",
        "3",
        "--descriptors-path",
        descriptors_path,
        "--atom-features-path",
        atom_features_path,
        "--bond-features-path",
        bond_features_path,
        "--atom-descriptors-path",
        atom_descriptors_path,
        "--bond-descriptors-path",
        bond_descriptors_path,
    ]


def test_predict_regression_quick(monkeypatch, regression_data_path, regression_model_path):
    input_path, *_ = regression_data_path
    args = ["chemprop", "predict", "-i", input_path, "--model-path", regression_model_path]

    with monkeypatch.context() as m:
        m.setattr("sys.argv", args)
        main()


def test_train_regression_no_mol(monkeypatch, regression_data_path):
    input_path, smiles, targets, atom_targets, bond_targets, weight = regression_data_path

    args = [
        "chemprop",
        "train",
        "-i",
        input_path,
        "--smiles-columns",
        smiles,
        "--atom-target-columns",
        *atom_targets,
        "--bond-target-columns",
        *bond_targets,
        "--epochs",
        "3",
    ]

    with monkeypatch.context() as m:
        m.setattr("sys.argv", args)
        main()


def test_predict_regression_no_mol(monkeypatch, regression_data_path, regression_no_mol_model_path):
    input_path, *_ = regression_data_path
    args = ["chemprop", "predict", "-i", input_path, "--model-path", regression_no_mol_model_path]

    with monkeypatch.context() as m:
        m.setattr("sys.argv", args)
        main()


def test_train_regresssion_no_atom(monkeypatch, regression_data_path):
    input_path, smiles, targets, atom_targets, bond_targets, weight = regression_data_path

    args = [
        "chemprop",
        "train",
        "-i",
        input_path,
        "--smiles-columns",
        smiles,
        "--target-columns",
        *targets,
        "--bond-target-columns",
        *bond_targets,
        "--epochs",
        "3",
    ]

    with monkeypatch.context() as m:
        m.setattr("sys.argv", args)
        main()


def test_predict_regression_no_atom(
    monkeypatch, regression_data_path, regression_no_atom_model_path
):
    input_path, *_ = regression_data_path
    args = ["chemprop", "predict", "-i", input_path, "--model-path", regression_no_atom_model_path]

    with monkeypatch.context() as m:
        m.setattr("sys.argv", args)
        main()


def test_train_regression_no_bond(monkeypatch, regression_data_path):
    input_path, smiles, targets, atom_targets, bond_targets, weight = regression_data_path

    args = [
        "chemprop",
        "train",
        "-i",
        input_path,
        "--smiles-columns",
        smiles,
        "--target-columns",
        *targets,
        "--atom-target-columns",
        *atom_targets,
        "--epochs",
        "3",
    ]

    with monkeypatch.context() as m:
        m.setattr("sys.argv", args)
        main()


def test_predict_regression_no_bond(
    monkeypatch, regression_data_path, regression_no_bond_model_path
):
    input_path, *_ = regression_data_path
    args = ["chemprop", "predict", "-i", input_path, "--model-path", regression_no_bond_model_path]

    with monkeypatch.context() as m:
        m.setattr("sys.argv", args)
        main()


# The test for only_mol is just the normal regression CLI tests


def test_train_regression_only_atom(monkeypatch, regression_data_path):
    input_path, smiles, targets, atom_targets, bond_targets, weight = regression_data_path

    args = [
        "chemprop",
        "train",
        "-i",
        input_path,
        "--smiles-columns",
        smiles,
        "--atom-target-columns",
        *atom_targets,
        "--epochs",
        "3",
    ]

    with monkeypatch.context() as m:
        m.setattr("sys.argv", args)
        main()


def test_predict_regression_only_atom(
    monkeypatch, regression_data_path, regression_only_atom_model_path
):
    input_path, *_ = regression_data_path
    args = [
        "chemprop",
        "predict",
        "-i",
        input_path,
        "--model-path",
        regression_only_atom_model_path,
    ]

    with monkeypatch.context() as m:
        m.setattr("sys.argv", args)
        main()


def test_train_regression_only_bond(monkeypatch, regression_data_path):
    input_path, smiles, targets, atom_targets, bond_targets, weight = regression_data_path

    args = [
        "chemprop",
        "train",
        "-i",
        input_path,
        "--smiles-columns",
        smiles,
        "--bond-target-columns",
        *bond_targets,
        "--epochs",
        "3",
    ]

    with monkeypatch.context() as m:
        m.setattr("sys.argv", args)
        main()


def test_predict_regression_only_bond(
    monkeypatch, regression_data_path, regression_only_bond_model_path
):
    input_path, *_ = regression_data_path
    args = [
        "chemprop",
        "predict",
        "-i",
        input_path,
        "--model-path",
        regression_only_bond_model_path,
    ]

    with monkeypatch.context() as m:
        m.setattr("sys.argv", args)
        main()


def test_train_bounded_quick(monkeypatch, bounded_data_path):
    input_path, smiles, targets, atom_targets, bond_targets, weight = bounded_data_path

    args = [
        "chemprop",
        "train",
        "-i",
        input_path,
        "--smiles-columns",
        smiles,
        "--target-columns",
        *targets,
        "--atom-target-columns",
        *atom_targets,
        "--bond-target-columns",
        *bond_targets,
        "--epochs",
        "3",
        "--loss-function",
        "bounded-mse",
    ]

    with monkeypatch.context() as m:
        m.setattr("sys.argv", args)
        main()


def test_train_classification_quick(monkeypatch, classification_data_path):
    input_path, smiles, targets, atom_targets, bond_targets, weight = classification_data_path

    args = [
        "chemprop",
        "train",
        "-i",
        input_path,
        "--smiles-columns",
        smiles,
        "--target-columns",
        *targets,
        "--atom-target-columns",
        *atom_targets,
        "--bond-target-columns",
        *bond_targets,
        "--epochs",
        "3",
        "--task-type",
        "classification",
    ]

    with monkeypatch.context() as m:
        m.setattr("sys.argv", args)
        main()


def test_predict_classification_quick(monkeypatch, regression_data_path, classification_model_path):
    input_path, *_ = regression_data_path
    args = ["chemprop", "predict", "-i", input_path, "--model-path", classification_model_path]

    with monkeypatch.context() as m:
        m.setattr("sys.argv", args)
        main()


def test_train_multiclass_quick(monkeypatch, multiclass_data_path):
    input_path, smiles, targets, atom_targets, bond_targets, weight = multiclass_data_path

    args = [
        "chemprop",
        "train",
        "-i",
        input_path,
        "--smiles-columns",
        smiles,
        "--target-columns",
        *targets,
        "--atom-target-columns",
        *atom_targets,
        "--bond-target-columns",
        *bond_targets,
        "--epochs",
        "3",
        "--task-type",
        "multiclass",
    ]

    with monkeypatch.context() as m:
        m.setattr("sys.argv", args)
        main()


def test_predict_multiclass_quick(monkeypatch, regression_data_path, multiclass_model_path):
    input_path, *_ = regression_data_path
    args = ["chemprop", "predict", "-i", input_path, "--model-path", multiclass_model_path]

    with monkeypatch.context() as m:
        m.setattr("sys.argv", args)
        main()


def test_train_mve_quick(monkeypatch, regression_data_path):
    input_path, smiles, targets, atom_targets, bond_targets, weight = regression_data_path

    args = [
        "chemprop",
        "train",
        "-i",
        input_path,
        "--smiles-columns",
        smiles,
        "--target-columns",
        *targets,
        "--atom-target-columns",
        *atom_targets,
        "--bond-target-columns",
        *bond_targets,
        "--epochs",
        "3",
        "--task-type",
        "regression-mve",
    ]

    with monkeypatch.context() as m:
        m.setattr("sys.argv", args)
        main()


def test_predict_mve_quick(monkeypatch, regression_data_path, mve_model_path):
    input_path, *_ = regression_data_path
    args = ["chemprop", "predict", "-i", input_path, "--model-path", mve_model_path]

    with monkeypatch.context() as m:
        m.setattr("sys.argv", args)
        main()


def test_train_regression_constrained(monkeypatch, constrained_data_path):
    (
        input_path,
        constraints_path,
        constraints_to_targets,
        smiles,
        targets,
        atom_targets,
        bond_targets,
    ) = constrained_data_path

    args = [
        "chemprop",
        "train",
        "-i",
        input_path,
        "--smiles-columns",
        smiles,
        "--target-columns",
        *targets,
        "--atom-target-columns",
        *atom_targets,
        "--bond-target-columns",
        *bond_targets,
        "--epochs",
        "3",
        "--constraints-path",
        constraints_path,
        "--constraints-to-targets",
        *constraints_to_targets,
    ]

    with monkeypatch.context() as m:
        m.setattr("sys.argv", args)
        main()


def test_predict_regression_constrained(monkeypatch, regression_data_path, constrained_model_path):
    input_path, *_ = regression_data_path
    args = ["chemprop", "predict", "-i", input_path, "--model-path", constrained_model_path]

    with monkeypatch.context() as m:
        m.setattr("sys.argv", args)
        main()


@pytest.mark.parametrize("ffn_block_index", ["0", "1"])
def test_fingerprint_quick(
    monkeypatch, regression_data_path, regression_model_path, ffn_block_index
):
    input_path, *_ = regression_data_path
    args = [
        "chemprop",
        "fingerprint",
        "-i",
        input_path,
        "--model-path",
        regression_model_path,
        "--ffn-block-index",
        ffn_block_index,
    ]

    with monkeypatch.context() as m:
        m.setattr("sys.argv", args)
        main()


@pytest.mark.skipif(NO_RAY or NO_OPTUNA, reason="Optuna not installed")
def test_optuna_quick(monkeypatch, regression_data_path, tmp_path):
    input_path, smiles, targets, atom_targets, bond_targets, weight = regression_data_path

    args = [
        "chemprop",
        "hpopt",
        "-i",
        input_path,
        "--smiles-columns",
        smiles,
        "--target-columns",
        *targets,
        "--atom-target-columns",
        *atom_targets,
        "--bond-target-columns",
        *bond_targets,
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
        "--accelerator",
        "cpu",
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
        "--is-mixed",
    ]

    with monkeypatch.context() as m:
        m.setattr("sys.argv", args)
        main()

    assert (tmp_path / "model_0" / "best.pt").exists()


@pytest.mark.skipif(NO_RAY or NO_HYPEROPT, reason="Ray and/or Hyperopt not installed")
def test_hyperopt_quick(monkeypatch, regression_data_path, tmp_path):
    input_path, smiles, targets, atom_targets, bond_targets, weight = regression_data_path

    args = [
        "chemprop",
        "hpopt",
        "-i",
        input_path,
        "--smiles-columns",
        smiles,
        "--target-columns",
        *targets,
        "--atom-target-columns",
        *atom_targets,
        "--bond-target-columns",
        *bond_targets,
        "--epochs",
        "6",
        "--hpopt-save-dir",
        str(tmp_path),
        "--raytune-num-samples",
        "2",
        "--raytune-search-algorithm",
        "hyperopt",
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
        "--is-mixed",
    ]

    with monkeypatch.context() as m:
        m.setattr("sys.argv", args)
        main()

    assert (tmp_path / "model_0" / "best.pt").exists()
