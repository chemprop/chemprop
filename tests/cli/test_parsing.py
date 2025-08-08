import numpy as np
import pandas as pd
import pytest

from chemprop.cli.main import main
from chemprop.cli.utils import build_MAB_data_from_files, make_dataset
from chemprop.data import MolAtomBondDatapoint, MolAtomBondDataset

# Maybe we will add a test for the normal parser in the future.


def test_MAB_parsing(data_dir):
    data = build_MAB_data_from_files(
        data_dir / "mol_atom_bond/regression.csv",
        ["smiles"],
        ["mol_y1", "mol_y2"],
        ["atom_y1", "atom_y2"],
        ["bond_y1", "bond_y2"],
        ["weight"],
        False,
        data_dir / "mol_atom_bond/descriptors.npz",
        {0: data_dir / "mol_atom_bond/atom_features_descriptors.npz"},
        {0: data_dir / "mol_atom_bond/atom_features_descriptors.npz"},
        {0: data_dir / "mol_atom_bond/bond_features_descriptors.npz"},
        {0: data_dir / "mol_atom_bond/bond_features_descriptors.npz"},
        None,
        None,
        molecule_featurizers=["morgan_binary"],
        keep_h=True,
        add_h=False,
        reorder_atoms=True,
        ignore_stereo=False,
    )
    data = data[0]
    assert len(data) == 11
    dp = data[0]
    assert isinstance(dp, MolAtomBondDatapoint)
    assert dp.y is not None
    assert dp.weight is not None
    assert dp.gt_mask is None
    assert dp.lt_mask is None
    assert dp.x_d is not None
    assert dp.name is not None
    assert dp.mol is not None
    assert dp.V_f is not None
    assert dp.V_d is not None
    assert dp.E_f is not None
    assert dp.E_d is not None
    assert dp.atom_y is not None
    assert dp.atom_gt_mask is None
    assert dp.atom_lt_mask is None
    assert dp.bond_y is not None
    assert dp.bond_gt_mask is None
    assert dp.bond_lt_mask is None
    assert dp.atom_constraint is None
    assert dp.bond_constraint is None

    dset = make_dataset(data)
    assert isinstance(dset, MolAtomBondDataset)


def test_MAB_parsing_bounded(data_dir):
    data = build_MAB_data_from_files(
        data_dir / "mol_atom_bond/bounded.csv",
        ["smiles"],
        ["mol_y1", "mol_y2"],
        ["atom_y1", "atom_y2"],
        ["bond_y1", "bond_y2"],
        ["weight"],
        True,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        keep_h=True,
        add_h=False,
        reorder_atoms=True,
        ignore_stereo=False,
    )
    data = data[0]
    assert len(data) == 11
    dp = data[0]
    assert isinstance(dp, MolAtomBondDatapoint)
    assert dp.y is not None
    assert dp.weight is not None
    assert dp.gt_mask is not None
    assert dp.lt_mask is not None
    assert dp.name is not None
    assert dp.mol is not None
    assert dp.atom_y is not None
    assert dp.atom_gt_mask is not None
    assert dp.atom_lt_mask is not None
    assert dp.bond_y is not None
    assert dp.bond_gt_mask is not None
    assert dp.bond_lt_mask is not None


def test_MAB_parsing_constrained(data_dir):
    data = build_MAB_data_from_files(
        data_dir / "mol_atom_bond/constrained_regression.csv",
        ["smiles"],
        ["mol_y"],
        ["atom_y1", "atom_y2"],
        ["bond_y1", "bond_y2"],
        None,
        False,
        None,
        None,
        None,
        None,
        None,
        data_dir / "mol_atom_bond/constrained_regression_constraints.csv",
        {"atom_target_col_0": 0, "atom_target_col_1": 1, "bond_target_col_1": 2},
        None,
        keep_h=True,
        add_h=False,
        reorder_atoms=True,
        ignore_stereo=False,
    )
    data = data[0]
    assert len(data) == 11
    dp = data[0]
    assert isinstance(dp, MolAtomBondDatapoint)
    assert dp.y is not None
    assert dp.weight is not None
    assert dp.name is not None
    assert dp.mol is not None
    assert dp.atom_y is not None
    assert dp.bond_y is not None
    assert dp.atom_constraint is not None
    assert dp.bond_constraint is not None


def test_MAB_parsing_constrained_error(data_dir):
    with pytest.raises(ValueError):
        build_MAB_data_from_files(
            data_dir / "mol_atom_bond/constrained_regression.csv",
            ["smiles"],
            ["mol_y"],
            ["atom_y1", "atom_y2"],
            ["bond_y1", "bond_y2"],
            None,
            False,
            None,
            None,
            None,
            None,
            None,
            data_dir / "mol_atom_bond/constrained_regression_constraints.csv",
            None,
            None,
            keep_h=True,
            add_h=False,
            reorder_atoms=True,
            ignore_stereo=False,
        )


def test_preds_stay_same(monkeypatch, tmp_path):
    args = [
        "chemprop",
        "predict",
        "-i",
        "tests/data/regression/rxn+mol/rxn+mol.csv",
        "--accelerator",
        "cpu",
        "--num-workers",
        "0",
        "--reaction-columns",
        "rxn_smiles",
        "--smiles-columns",
        "solvent_smiles",
        "solvent_smiles",
        "--atom-features-path",
        "1",
        "tests/data/regression/rxn+mol/atom_features.npz",
        "--no-atom-feature-scaling",
        "--atom-descriptors-path",
        "tests/data/regression/rxn+mol/atom_descriptors.npz",
        "--bond-features-path",
        "tests/data/regression/rxn+mol/bond_features.npz",
        "--descriptors-path",
        "tests/data/regression/rxn+mol/descriptors.npz",
        "--rxn-mode",
        "REAC_DIFF_BALANCE",
        "--multi-hot-atom-featurizer-mode",
        "RIGR",
        "--keep-h",
        "--molecule-featurizers",
        "morgan_count",
        "--model-path",
        "tests/data/example_model_v2_regression_mol+mol+rxn_check_predictions.pt",
        "-o",
        str(tmp_path / "preds.csv"),
    ]

    with monkeypatch.context() as m:
        m.setattr("sys.argv", args)
        main()

    preds_df = pd.read_csv(tmp_path / "preds.csv")
    expected_df = pd.read_pickle("tests/data/data_for_test_preds_stay_same.pkl")

    assert np.allclose(
        preds_df["target"].values, expected_df["target"].values, rtol=1e-4, atol=1e-5
    )
