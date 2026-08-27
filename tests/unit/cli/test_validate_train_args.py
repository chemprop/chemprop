from configargparse import ArgumentError, ArgumentParser
import numpy as np
import pandas as pd
import pytest

from chemprop.cli.common import process_common_args, validate_common_args
from chemprop.cli.train import (
    TrainSubcommand,
    process_train_args,
    validate_train_args,
)


@pytest.mark.parametrize(
    "feature_flag",
    [
        "--descriptors-path",
        "--atom-features-path",
        "--atom-descriptors-path",
        "--bond-features-path",
        "--bond-descriptors-path",
    ],
)
def test_extra_features_with_separate_data_files(tmp_path, feature_flag):
    data_paths = []
    for split, smiles in zip(("train", "val", "test"), ("C", "CC", "CCC")):
        data_path = tmp_path / f"{split}.csv"
        pd.DataFrame({"smiles": [smiles], "target": [0.0]}).to_csv(data_path, index=False)
        data_paths.append(data_path)

    features_path = tmp_path / "features.npz"
    np.savez(features_path, np.array([[1.0], [2.0], [3.0]]))

    parser = TrainSubcommand.add_args(ArgumentParser())
    args = parser.parse_args(
        [
            "--data-path",
            *map(str, data_paths),
            feature_flag,
            str(features_path),
            "--output-dir",
            str(tmp_path / "output"),
        ]
    )
    args = process_common_args(args)
    validate_common_args(args)
    args = process_train_args(args)

    with pytest.raises(ArgumentError, match=f"{feature_flag}.*separate data files"):
        validate_train_args(args)
