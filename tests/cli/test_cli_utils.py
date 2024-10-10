import pytest

from chemprop.cli.common import find_models
from chemprop.cli.utils.parsing import get_column_names, parse_indices


def test_parse_indices():
    """
    Testing if CLI parse_indices yields expected results.
    """
    splits = {"train": [0, 1, 2, 4], "val": [3, 5, 6], "test": [7, 8, 9]}
    split_idxs = {"train": "0-2, 4", "val": "3,5-6", "test": [7, 8, 9]}
    split_idxs = {split: parse_indices(idxs) for split, idxs in split_idxs.items()}

    assert split_idxs == splits


def test_find_models(data_dir):
    """
    Testing if CLI find_models gets the correct model paths.
    """
    models = find_models([data_dir / "example_model_v2_regression_mol.pt"])
    assert len(models) == 1
    models = find_models([data_dir / "example_model_v2_regression_mol.ckpt"])
    assert len(models) == 1
    models = find_models([data_dir])
    assert len(models) == 10
    models = find_models(
        [
            data_dir / "example_model_v2_regression_mol.pt",
            data_dir / "example_model_v2_regression_mol.ckpt",
            data_dir,
        ]
    )
    assert len(models) == 12


@pytest.mark.parametrize(
    "path,smiles_cols,rxn_cols,target_cols,ignore_cols,splits_col,weight_col,no_header_row,expected",
    [
        (
            "classification/mol.csv",
            ["smiles"],
            None,
            ["NR-AhR", "NR-ER", "SR-ARE", "SR-MMP"],
            None,
            None,
            None,
            False,
            ["smiles", "NR-AhR", "NR-ER", "SR-ARE", "SR-MMP"],
        ),
        (
            "classification/mol.csv",
            ["smiles"],
            None,
            None,
            None,
            None,
            None,
            False,
            ["smiles", "NR-AhR", "NR-ER", "SR-ARE", "SR-MMP"],
        ),
        (
            "classification/mol.csv",
            None,
            None,
            None,
            ["NR-AhR", "SR-ARE"],
            None,
            None,
            False,
            ["smiles", "NR-ER", "SR-MMP"],
        ),
        ("regression/mol/mol.csv", None, None, None, None, None, None, False, ["smiles", "lipo"]),
        (
            "regression/mol/mol.csv",
            None,
            None,
            ["lipo"],
            None,
            None,
            None,
            False,
            ["smiles", "lipo"],
        ),
        (
            "regression/mol/mol_with_splits.csv",
            ["smiles"],
            None,
            ["lipo"],
            None,
            ["split"],
            None,
            False,
            ["smiles", "lipo"],
        ),
        (
            "regression/mol/mol_with_splits.csv",
            None,
            None,
            None,
            None,
            ["split"],
            None,
            False,
            ["smiles", "lipo"],
        ),
        (
            "regression/rxn/rxn.csv",
            None,
            ["smiles"],
            ["ea"],
            None,
            None,
            None,
            False,
            ["smiles", "ea"],
        ),
        (
            "classification/mol+mol.csv",
            ["mol a smiles", "mol b Smiles"],
            None,
            ["synergy"],
            None,
            None,
            None,
            False,
            ["mol a smiles", "mol b Smiles", "synergy"],
        ),
        (
            "classification/mol+mol.csv",
            ["mol a smiles", "mol b Smiles"],
            None,
            None,
            None,
            None,
            None,
            False,
            ["mol a smiles", "mol b Smiles", "synergy"],
        ),
        ("regression/mol/mol.csv", None, None, None, None, None, None, True, ["SMILES", "pred_0"]),
    ],
)
def test_get_column_names(
    data_dir,
    path,
    smiles_cols,
    rxn_cols,
    target_cols,
    ignore_cols,
    splits_col,
    weight_col,
    no_header_row,
    expected,
):
    """
    Testing if CLI get_column_names gets the correct column names.
    """
    input_cols, target_cols = get_column_names(
        data_dir / path,
        smiles_cols,
        rxn_cols,
        target_cols,
        ignore_cols,
        splits_col,
        weight_col,
        no_header_row,
    )

    assert input_cols + target_cols == expected
