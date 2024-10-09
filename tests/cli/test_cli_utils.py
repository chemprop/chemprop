from chemprop.cli.common import find_models
from chemprop.cli.utils.parsing import parse_indices


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
