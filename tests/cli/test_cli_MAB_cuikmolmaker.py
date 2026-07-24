"""Tests for `--use-cuikmolmaker-featurization` combined with MolAtomBond (atom/bond target)
training and prediction.
"""
import pandas as pd
import pytest

from chemprop.cli.main import main

pytestmark = pytest.mark.CLI

# `regression.csv` and `constrained_regression.csv` both include a `[H][H]` datapoint (which
# requires `--keep-h`) and an atom-mapped datapoint (which requires `--reorder-atoms`). Both flags
# are incompatible with `--use-cuikmolmaker-featurization`, so those rows are filtered out here.
_INCOMPATIBLE_SMIS = ["[H][H]", "[CH2:3]=[N+:1]([H:4])[H:2]"]


@pytest.fixture
def cuik_regression_data_path(data_dir, tmp_path):
    src = data_dir / "mol_atom_bond" / "regression.csv"
    df = pd.read_csv(src)
    df = df[~df["smiles"].isin(_INCOMPATIBLE_SMIS)]
    dst = tmp_path / "regression_cuik.csv"
    df.to_csv(dst, index=False)

    return (
        str(dst),
        "smiles",
        ["mol_y1", "mol_y2"],
        ["atom_y1", "atom_y2"],
        ["bond_y1", "bond_y2"],
        "weight",
    )


@pytest.fixture
def cuik_constrained_data_path(data_dir, tmp_path):
    src = data_dir / "mol_atom_bond" / "constrained_regression.csv"
    src_constraints = data_dir / "mol_atom_bond" / "constrained_regression_constraints.csv"
    df = pd.read_csv(src)
    df_constraints = pd.read_csv(src_constraints)

    keep_mask = ~df["smiles"].isin(_INCOMPATIBLE_SMIS)
    df = df[keep_mask]
    df_constraints = df_constraints[keep_mask.to_numpy()]

    dst = tmp_path / "constrained_regression_cuik.csv"
    dst_constraints = tmp_path / "constrained_regression_constraints_cuik.csv"
    df.to_csv(dst, index=False)
    df_constraints.to_csv(dst_constraints, index=False)

    return (
        str(dst),
        str(dst_constraints),
        ["atom_y1", "atom_y2", "bond_y2"],
        "smiles",
        ["mol_y"],
        ["atom_y1", "atom_y2"],
        ["bond_y1", "bond_y2"],
    )


def test_train_quick_cuikmolmaker(monkeypatch, cuik_regression_data_path):
    input_path, smiles, mol_targets, atom_targets, bond_targets, weight = cuik_regression_data_path

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
        "--use-cuikmolmaker-featurization",
        "--split-sizes",
        "0.4",
        "0.3",
        "0.3",
    ]

    with monkeypatch.context() as m:
        m.setattr("sys.argv", args)
        main()


def test_train_predict_quick_cuikmolmaker(monkeypatch, cuik_regression_data_path, tmp_path):
    input_path, smiles, mol_targets, atom_targets, bond_targets, weight = cuik_regression_data_path
    save_dir = tmp_path / "cuik_mab_model"

    train_args = [
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
        "--use-cuikmolmaker-featurization",
        "--save-dir",
        str(save_dir),
        "--split-sizes",
        "0.4",
        "0.3",
        "0.3",
    ]

    with monkeypatch.context() as m:
        m.setattr("sys.argv", train_args)
        main()

    model_path = save_dir / "model_0" / "best.pt"
    assert model_path.exists()

    predict_args = [
        "chemprop",
        "predict",
        "-i",
        input_path,
        "--smiles-columns",
        smiles,
        "--model-paths",
        str(model_path),
        "--use-cuikmolmaker-featurization",
        "--output",
        str(tmp_path / "cuik_mab_preds.csv"),
    ]

    with monkeypatch.context() as m:
        m.setattr("sys.argv", predict_args)
        main()

    assert (tmp_path / "cuik_mab_preds.csv").exists()


def test_train_only_atom_cuikmolmaker(monkeypatch, cuik_regression_data_path):
    input_path, smiles, mol_targets, atom_targets, bond_targets, weight = cuik_regression_data_path

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
        "--use-cuikmolmaker-featurization",
        "--split-sizes",
        "0.4",
        "0.3",
        "0.3",
    ]

    with monkeypatch.context() as m:
        m.setattr("sys.argv", args)
        main()


def test_train_constrained_quick_cuikmolmaker(monkeypatch, cuik_constrained_data_path):
    (
        input_path,
        constraints_path,
        constraints_to_targets,
        smiles,
        mol_targets,
        atom_targets,
        bond_targets,
    ) = cuik_constrained_data_path

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
        "--constraints-path",
        constraints_path,
        "--constraints-to-targets",
        *constraints_to_targets,
        "--epochs",
        "3",
        "--use-cuikmolmaker-featurization",
        "--split-sizes",
        "0.4",
        "0.3",
        "0.3",
    ]

    with monkeypatch.context() as m:
        m.setattr("sys.argv", args)
        main()
