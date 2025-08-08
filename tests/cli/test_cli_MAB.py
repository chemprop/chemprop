"""This tests the CLI functionality of training and predicting a regression model on a single molecule.
"""

import numpy as np
import pytest

from chemprop.cli.hpopt import NO_HYPEROPT, NO_OPTUNA, NO_RAY
from chemprop.cli.main import main
from chemprop.cli.utils.MAB_parsing import build_MAB_data_from_files

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
        ["atom_y1", "atom_y2", "bond_y2"],
        "smiles",
        ["mol_y"],
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
def regression_extras_model_path(model_dir):
    return str(model_dir / "regression_with_extras.pt")


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
def regression_only_mol_model_path(model_dir):
    return str(model_dir / "regression_only_mol.pt")


@pytest.fixture
def regression_only_atom_model_path(model_dir):
    return str(model_dir / "regression_only_atom.pt")


@pytest.fixture
def regression_only_bond_model_path(model_dir):
    return str(model_dir / "regression_only_bond.pt")


@pytest.fixture
def classification_model_path(model_dir):
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
    input_path, smiles, mol_targets, atom_targets, bond_targets, weight = regression_data_path

    args = [
        "chemprop",
        "train",
        "-i",
        input_path,
        "--smiles-columns",
        smiles,
        "--mol-target-columns",
        *mol_targets,
        "--atom-target-columns",
        *atom_targets,
        "--bond-target-columns",
        *bond_targets,
        "--weight-column",
        weight,
        "--epochs",
        "3",
        "--show-individual-scores",
        "--keep-h",
        "--reorder-atoms",
        "--split-sizes",
        "0.4",
        "0.3",
        "0.3",
    ]

    with monkeypatch.context() as m:
        m.setattr("sys.argv", args)
        main()


def test_train_regression_quick_features(monkeypatch, regression_data_path, extras_paths):
    input_path, smiles, mol_targets, atom_targets, bond_targets, weight = regression_data_path
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
        "--mol-target-columns",
        *mol_targets,
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
        "--keep-h",
        "--reorder-atoms",
        "--split-sizes",
        "0.4",
        "0.3",
        "0.3",
    ]

    with monkeypatch.context() as m:
        m.setattr("sys.argv", args)
        main()


def test_predict_regression_quick_features(
    monkeypatch, regression_data_path, extras_paths, regression_extras_model_path
):
    input_path, smiles, mol_targets, atom_targets, bond_targets, weight = regression_data_path
    (
        descriptors_path,
        atom_features_path,
        bond_features_path,
        atom_descriptors_path,
        bond_descriptors_path,
    ) = extras_paths

    args = [
        "chemprop",
        "predict",
        "-i",
        input_path,
        "--model-path",
        regression_extras_model_path,
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
        "--keep-h",
        "--reorder-atoms",
    ]

    with monkeypatch.context() as m:
        m.setattr("sys.argv", args)
        main()


def test_predict_regression_quick(monkeypatch, regression_data_path, regression_model_path):
    input_path, *_ = regression_data_path
    args = [
        "chemprop",
        "predict",
        "-i",
        input_path,
        "--model-path",
        regression_model_path,
        "--keep-h",
        "--reorder-atoms",
    ]

    with monkeypatch.context() as m:
        m.setattr("sys.argv", args)
        main()


def test_train_regression_no_mol(monkeypatch, regression_data_path):
    input_path, smiles, mol_targets, atom_targets, bond_targets, weight = regression_data_path

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
        "--keep-h",
        "--reorder-atoms",
        "--split-sizes",
        "0.4",
        "0.3",
        "0.3",
    ]

    with monkeypatch.context() as m:
        m.setattr("sys.argv", args)
        main()


def test_predict_regression_no_mol(monkeypatch, regression_data_path, regression_no_mol_model_path):
    input_path, *_ = regression_data_path
    args = [
        "chemprop",
        "predict",
        "-i",
        input_path,
        "--model-path",
        regression_no_mol_model_path,
        "--keep-h",
        "--reorder-atoms",
    ]

    with monkeypatch.context() as m:
        m.setattr("sys.argv", args)
        main()


def test_train_regresssion_no_atom(monkeypatch, regression_data_path):
    input_path, smiles, mol_targets, atom_targets, bond_targets, weight = regression_data_path

    args = [
        "chemprop",
        "train",
        "-i",
        input_path,
        "--smiles-columns",
        smiles,
        "--mol-target-columns",
        *mol_targets,
        "--bond-target-columns",
        *bond_targets,
        "--epochs",
        "3",
        "--keep-h",
        "--reorder-atoms",
        "--split-sizes",
        "0.4",
        "0.3",
        "0.3",
    ]

    with monkeypatch.context() as m:
        m.setattr("sys.argv", args)
        main()


def test_predict_regression_no_atom(
    monkeypatch, regression_data_path, regression_no_atom_model_path
):
    input_path, *_ = regression_data_path
    args = [
        "chemprop",
        "predict",
        "-i",
        input_path,
        "--model-path",
        regression_no_atom_model_path,
        "--keep-h",
        "--reorder-atoms",
    ]

    with monkeypatch.context() as m:
        m.setattr("sys.argv", args)
        main()


def test_train_regression_no_bond(monkeypatch, regression_data_path):
    input_path, smiles, mol_targets, atom_targets, bond_targets, weight = regression_data_path

    args = [
        "chemprop",
        "train",
        "-i",
        input_path,
        "--smiles-columns",
        smiles,
        "--mol-target-columns",
        *mol_targets,
        "--atom-target-columns",
        *atom_targets,
        "--epochs",
        "3",
        "--keep-h",
        "--reorder-atoms",
        "--split-sizes",
        "0.4",
        "0.3",
        "0.3",
    ]

    with monkeypatch.context() as m:
        m.setattr("sys.argv", args)
        main()


def test_predict_regression_no_bond(
    monkeypatch, regression_data_path, regression_no_bond_model_path
):
    input_path, *_ = regression_data_path
    args = [
        "chemprop",
        "predict",
        "-i",
        input_path,
        "--model-path",
        regression_no_bond_model_path,
        "--keep-h",
        "--reorder-atoms",
    ]

    with monkeypatch.context() as m:
        m.setattr("sys.argv", args)
        main()


def test_train_regression_only_mol(monkeypatch, regression_data_path):
    input_path, smiles, mol_targets, atom_targets, bond_targets, weight = regression_data_path

    args = [
        "chemprop",
        "train",
        "-i",
        input_path,
        "--smiles-columns",
        smiles,
        "--mol-target-columns",
        *mol_targets,
        "--epochs",
        "3",
        "--keep-h",
        "--reorder-atoms",
        "--split-sizes",
        "0.4",
        "0.3",
        "0.3",
    ]

    with monkeypatch.context() as m:
        m.setattr("sys.argv", args)
        main()


def test_predict_regression_only_mol(
    monkeypatch, regression_data_path, regression_only_mol_model_path
):
    input_path, *_ = regression_data_path
    args = [
        "chemprop",
        "predict",
        "-i",
        input_path,
        "--model-path",
        regression_only_mol_model_path,
        "--keep-h",
        "--reorder-atoms",
    ]

    with monkeypatch.context() as m:
        m.setattr("sys.argv", args)
        main()


def test_train_regression_only_atom(monkeypatch, regression_data_path):
    input_path, smiles, mol_targets, atom_targets, bond_targets, weight = regression_data_path

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
        "--keep-h",
        "--reorder-atoms",
        "--split-sizes",
        "0.4",
        "0.3",
        "0.3",
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
        "--keep-h",
        "--reorder-atoms",
    ]

    with monkeypatch.context() as m:
        m.setattr("sys.argv", args)
        main()


def test_train_regression_only_bond(monkeypatch, regression_data_path):
    input_path, smiles, mol_targets, atom_targets, bond_targets, weight = regression_data_path

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
        "--keep-h",
        "--reorder-atoms",
        "--split-sizes",
        "0.4",
        "0.3",
        "0.3",
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
        "--keep-h",
        "--reorder-atoms",
    ]

    with monkeypatch.context() as m:
        m.setattr("sys.argv", args)
        main()


def test_train_bounded_quick(monkeypatch, bounded_data_path):
    input_path, smiles, mol_targets, atom_targets, bond_targets, weight = bounded_data_path

    args = [
        "chemprop",
        "train",
        "-i",
        input_path,
        "--smiles-columns",
        smiles,
        "--mol-target-columns",
        *mol_targets,
        "--atom-target-columns",
        *atom_targets,
        "--bond-target-columns",
        *bond_targets,
        "--epochs",
        "3",
        "--loss-function",
        "bounded-mse",
        "--keep-h",
        "--reorder-atoms",
        "--split-sizes",
        "0.4",
        "0.3",
        "0.3",
    ]

    with monkeypatch.context() as m:
        m.setattr("sys.argv", args)
        main()


def test_train_classification_quick(monkeypatch, classification_data_path):
    input_path, smiles, mol_targets, atom_targets, bond_targets, weight = classification_data_path

    args = [
        "chemprop",
        "train",
        "-i",
        input_path,
        "--smiles-columns",
        smiles,
        "--mol-target-columns",
        *mol_targets,
        "--atom-target-columns",
        *atom_targets,
        "--bond-target-columns",
        *bond_targets,
        "--epochs",
        "3",
        "--task-type",
        "classification",
        "--keep-h",
        "--reorder-atoms",
        "--split-sizes",
        "0.4",
        "0.3",
        "0.3",
    ]

    with monkeypatch.context() as m:
        m.setattr("sys.argv", args)
        main()


def test_predict_classification_quick(monkeypatch, regression_data_path, classification_model_path):
    input_path, *_ = regression_data_path
    args = [
        "chemprop",
        "predict",
        "-i",
        input_path,
        "--model-path",
        classification_model_path,
        "--keep-h",
        "--reorder-atoms",
    ]

    with monkeypatch.context() as m:
        m.setattr("sys.argv", args)
        main()


def test_train_multiclass_quick(monkeypatch, multiclass_data_path):
    input_path, smiles, mol_targets, atom_targets, bond_targets, weight = multiclass_data_path

    args = [
        "chemprop",
        "train",
        "-i",
        input_path,
        "--smiles-columns",
        smiles,
        "--mol-target-columns",
        *mol_targets,
        "--atom-target-columns",
        *atom_targets,
        "--bond-target-columns",
        *bond_targets,
        "--epochs",
        "3",
        "--task-type",
        "multiclass",
        "--keep-h",
        "--reorder-atoms",
        "--split-sizes",
        "0.4",
        "0.3",
        "0.3",
    ]

    with monkeypatch.context() as m:
        m.setattr("sys.argv", args)
        main()


def test_predict_multiclass_quick(monkeypatch, regression_data_path, multiclass_model_path):
    input_path, *_ = regression_data_path
    args = [
        "chemprop",
        "predict",
        "-i",
        input_path,
        "--model-path",
        multiclass_model_path,
        "--keep-h",
        "--reorder-atoms",
    ]

    with monkeypatch.context() as m:
        m.setattr("sys.argv", args)
        main()


def test_train_mve_quick(monkeypatch, regression_data_path):
    input_path, smiles, mol_targets, atom_targets, bond_targets, weight = regression_data_path

    args = [
        "chemprop",
        "train",
        "-i",
        input_path,
        "--smiles-columns",
        smiles,
        "--mol-target-columns",
        *mol_targets,
        "--atom-target-columns",
        *atom_targets,
        "--bond-target-columns",
        *bond_targets,
        "--epochs",
        "3",
        "--task-type",
        "regression-mve",
        "--keep-h",
        "--reorder-atoms",
        "--split-sizes",
        "0.4",
        "0.3",
        "0.3",
    ]

    with monkeypatch.context() as m:
        m.setattr("sys.argv", args)
        main()


def test_predict_mve_quick(monkeypatch, regression_data_path, mve_model_path):
    input_path, smiles, mol_targets, atom_targets, bond_targets, weight = regression_data_path
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
        "--keep-h",
        "--reorder-atoms",
    ]

    with monkeypatch.context() as m:
        m.setattr("sys.argv", args)
        main()


def test_train_regression_constrained(monkeypatch, constrained_data_path):
    (
        input_path,
        constraints_path,
        constraints_to_targets,
        smiles,
        mol_targets,
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
        "--mol-target-columns",
        *mol_targets,
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
        "--keep-h",
        "--reorder-atoms",
        "--split-sizes",
        "0.4",
        "0.3",
        "0.3",
    ]

    with monkeypatch.context() as m:
        m.setattr("sys.argv", args)
        main()


def test_predict_regression_constrained(monkeypatch, constrained_data_path, constrained_model_path):
    (input_path, constraints_path, constraints_to_targets, smiles, *_) = constrained_data_path

    args = [
        "chemprop",
        "predict",
        "-i",
        input_path,
        "--smiles-columns",
        smiles,
        "--model-path",
        constrained_model_path,
        "--constraints-path",
        constraints_path,
        "--constraints-to-targets",
        *constraints_to_targets,
        "--keep-h",
        "--reorder-atoms",
    ]

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
        "--keep-h",
        "--reorder-atoms",
    ]

    with monkeypatch.context() as m:
        m.setattr("sys.argv", args)
        main()


@pytest.mark.skipif(NO_RAY or NO_OPTUNA, reason="Optuna not installed")
def test_optuna_quick(monkeypatch, regression_data_path, tmp_path):
    input_path, smiles, mol_targets, atom_targets, bond_targets, weight = regression_data_path

    args = [
        "chemprop",
        "hpopt",
        "-i",
        input_path,
        "--smiles-columns",
        smiles,
        "--mol-target-columns",
        *mol_targets,
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
        "--keep-h",
        "--reorder-atoms",
        "--split-sizes",
        "0.4",
        "0.3",
        "0.3",
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
def test_hyperopt_quick(monkeypatch, constrained_data_path, tmp_path):
    (
        input_path,
        constraints_path,
        constraints_to_targets,
        smiles,
        mol_targets,
        atom_targets,
        bond_targets,
    ) = constrained_data_path

    args = [
        "chemprop",
        "hpopt",
        "-i",
        input_path,
        "--smiles-columns",
        smiles,
        "--mol-target-columns",
        *mol_targets,
        "--atom-target-columns",
        *atom_targets,
        "--bond-target-columns",
        *bond_targets,
        "--constraints-path",
        constraints_path,
        "--constraints-to-targets",
        *constraints_to_targets,
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
        "--keep-h",
        "--reorder-atoms",
        "--split-sizes",
        "0.4",
        "0.3",
        "0.3",
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
        "--keep-h",
        "--reorder-atoms",
    ]

    with monkeypatch.context() as m:
        m.setattr("sys.argv", args)
        main()

    assert (tmp_path / "model_0" / "best.pt").exists()


def test_make_predictions_with_constraints(monkeypatch, data_dir, tmpdir):
    constraint_cols = [
        "hirshfeld_charges",
        "hirshfeld_charges_plus1",
        "hirshfeld_charges_minus1",
        "hirshfeld_spin_density_plus1",
        "hirshfeld_spin_density_minus1",
        "hirshfeld_charges_fukui_neu",
        "hirshfeld_charges_fukui_elec",
    ]
    args = [
        "chemprop",
        "predict",
        "--model-path",
        str(data_dir / "mol_atom_bond" / "example_models" / "QM_descriptors.pt"),
        "-i",
        str(data_dir / "mol_atom_bond" / "atomic_bond_regression.csv"),
        "--constraints-to-targets",
        *constraint_cols,
        "--constraints-path",
        str(data_dir / "mol_atom_bond" / "atomic_bond_constraints.csv"),
        "-o",
        str(tmpdir / "preds.csv"),
        "--add-h",
    ]
    with monkeypatch.context() as m:
        m.setattr("sys.argv", args)
        main()

    old_data = build_MAB_data_from_files(
        p_data=data_dir / "mol_atom_bond" / "atomic_bond_regression_preds.csv",
        smiles_cols=["smiles"],
        mol_target_cols=["homo.1", "lumo.1"],
        atom_target_cols=[
            "hirshfeld_charges.1",
            "hirshfeld_charges_plus1.1",
            "hirshfeld_charges_minus1.1",
            "hirshfeld_spin_density_plus1.1",
            "hirshfeld_spin_density_minus1.1",
            "hirshfeld_charges_fukui_neu.1",
            "hirshfeld_charges_fukui_elec.1",
            "NMR.1",
        ],
        bond_target_cols=["bond_length_matrix.1", "bond_index_matrix.1"],
        weight_col=None,
        bounded=False,
        p_descriptors=None,
        p_atom_feats=None,
        p_bond_feats=None,
        p_atom_descs=None,
        p_bond_descs=None,
        p_constraints=None,
        constraints_cols_to_target_cols=None,
        molecule_featurizers=None,
        add_h=True,
    )
    new_data = build_MAB_data_from_files(
        p_data=tmpdir / "preds.csv",
        smiles_cols=["smiles"],
        mol_target_cols=["homo.1", "lumo.1"],
        atom_target_cols=[
            "hirshfeld_charges.1",
            "hirshfeld_charges_plus1.1",
            "hirshfeld_charges_minus1.1",
            "hirshfeld_spin_density_plus1.1",
            "hirshfeld_spin_density_minus1.1",
            "hirshfeld_charges_fukui_neu.1",
            "hirshfeld_charges_fukui_elec.1",
            "NMR.1",
        ],
        bond_target_cols=["bond_length_matrix.1", "bond_index_matrix.1"],
        weight_col=None,
        bounded=False,
        p_descriptors=None,
        p_atom_feats=None,
        p_bond_feats=None,
        p_atom_descs=None,
        p_bond_descs=None,
        p_constraints=None,
        constraints_cols_to_target_cols=None,
        molecule_featurizers=None,
        add_h=True,
    )
    old_data, new_data = old_data[0], new_data[0]

    for i in range(len(old_data)):
        np.testing.assert_allclose(old_data[i].y, new_data[i].y, atol=1e-4, rtol=1e-6)
        np.testing.assert_allclose(old_data[i].atom_y, new_data[i].atom_y, atol=1e-4, rtol=1e-6)
        np.testing.assert_allclose(old_data[i].bond_y, new_data[i].bond_y, atol=1e-4, rtol=1e-6)


def test_make_predictions_with_atom_map(monkeypatch, data_dir, tmpdir):
    args = [
        "chemprop",
        "predict",
        "--model-path",
        str(data_dir / "mol_atom_bond" / "example_models" / "atomic_regression_atom_mapped.pt"),
        "-i",
        str(data_dir / "mol_atom_bond" / "atomic_regression_atom_mapped.csv"),
        "-o",
        str(tmpdir / "preds.csv"),
        "--keep-h",
        "--reorder-atoms",
    ]
    with monkeypatch.context() as m:
        m.setattr("sys.argv", args)
        main()

    old_data = build_MAB_data_from_files(
        p_data=data_dir / "mol_atom_bond" / "atomic_regression_atom_mapped_preds.csv",
        smiles_cols=["smiles"],
        mol_target_cols=None,
        atom_target_cols=["charges"],
        bond_target_cols=None,
        weight_col=None,
        bounded=False,
        p_descriptors=None,
        p_atom_feats=None,
        p_bond_feats=None,
        p_atom_descs=None,
        p_bond_descs=None,
        p_constraints=None,
        constraints_cols_to_target_cols=None,
        molecule_featurizers=None,
        keep_h=True,
        reorder_atoms=True,
    )
    new_data = build_MAB_data_from_files(
        p_data=tmpdir / "preds.csv",
        smiles_cols=["smiles"],
        mol_target_cols=None,
        atom_target_cols=["charges"],
        bond_target_cols=None,
        weight_col=None,
        bounded=False,
        p_descriptors=None,
        p_atom_feats=None,
        p_bond_feats=None,
        p_atom_descs=None,
        p_bond_descs=None,
        p_constraints=None,
        constraints_cols_to_target_cols=None,
        molecule_featurizers=None,
        keep_h=True,
        reorder_atoms=True,
    )
    old_data, new_data = old_data[0], new_data[0]

    for i in range(len(old_data)):
        np.testing.assert_allclose(old_data[i].atom_y, new_data[i].atom_y, atol=1e-4, rtol=1e-6)
