from configargparse import ArgumentError, ArgumentParser
import numpy as np
import pandas as pd
import pytest

from chemprop.cli.common import process_common_args, validate_common_args
from chemprop.cli.train import TrainSubcommand, process_train_args, validate_train_args


@pytest.mark.parametrize(
    ("external_data_flag", "file_type"),
    [
        ("--descriptors-path", "npz"),
        ("--atom-features-path", "npz"),
        ("--atom-descriptors-path", "npz"),
        ("--bond-features-path", "npz"),
        ("--bond-descriptors-path", "npz"),
        ("--constraints-path", "csv"),
    ],
)
@pytest.mark.parametrize("num_data_paths", [2, 3])
def test_external_data_with_separate_data_files(
    tmp_path, external_data_flag, file_type, num_data_paths
):
    data_paths = []
    for split, smiles in zip(
        ("train", "val", "test")[:num_data_paths], ("C", "CC", "CCC")[:num_data_paths]
    ):
        data_path = tmp_path / f"{split}.csv"
        pd.DataFrame({"smiles": [smiles], "target": [0.0]}).to_csv(data_path, index=False)
        data_paths.append(data_path)

    external_data_path = tmp_path / f"external_data.{file_type}"
    if file_type == "npz":
        np.savez(external_data_path, np.array([[1.0], [2.0], [3.0]]))
    else:
        pd.DataFrame({"constraint": [1.0, 2.0, 3.0]}).to_csv(external_data_path, index=False)

    parser = TrainSubcommand.add_args(ArgumentParser())
    args = parser.parse_args(
        [
            "--data-path",
            *map(str, data_paths),
            external_data_flag,
            str(external_data_path),
            "--split-sizes",
            "0.8",
            "0.2",
            "0.0",
            "--output-dir",
            str(tmp_path / "output"),
        ]
    )
    args = process_common_args(args)
    validate_common_args(args)
    args = process_train_args(args)

    with pytest.raises(ArgumentError, match=f"{external_data_flag}.*separate data files"):
        validate_train_args(args)
