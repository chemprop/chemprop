"""Chemprop integration tests for feature generation."""
import os
from tempfile import TemporaryDirectory
from typing import List
import pandas as pd
import pytest
from parameterized import parameterized

from chemprop.constants import TEST_SCORES_FILE_NAME

import utils

TEST_DATA_DIR = "tests/data"
SEED = 0
EPOCHS = 3
NUM_FOLDS = 3
NUM_ITER = 2
DELTA = 0.015


@parameterized.expand(
    [
        (
            "chemprop_morgan_features_generator",
            "chemprop",
            "auc",
            ["--features_generator", "morgan"],
        ),
        (
            "chemprop_rdkit2d_features_generator",
            "chemprop",
            "auc",
            ["--features_generator", "rdkit_2d"],
        ),
        (
            "chemprop_morgan_count_features_generator",
            "chemprop",
            "auc",
            ["--features_generator", "morgan_count"],
        ),
        (
            "chemprop_rdkit_2d_normalized_count_features_generator",
            "chemprop",
            "auc",
            [
                "--features_generator",
                "rdkit_2d_normalized",
                "--no_features_scaling",
            ],
        ),
        (
            "chemprop_combined_features_generator",
            "chemprop",
            "auc",
            ["--features_generator", "rdkit_2d", "morgan"],
        ),
        (
            "chemprop_combined_features_generator_multimolecule",
            "chemprop",
            "auc",
            [
                "--features_generator",
                "rdkit_2d",
                "morgan",
                "--number_of_molecules",
                "2",
                "--data_path",
                os.path.join(TEST_DATA_DIR, "classification_multimolecule.csv"),
            ],
        ),
    ]
)
def test_feature_precomputation(
    self, name: str, model_type: str, metric: str, train_flags: List[str] = None
):
    with TemporaryDirectory() as save_dir:
        # Train with unbatched generators
        utils.train(
            dataset_type="classification",
            metric=metric,
            save_dir=save_dir,
            model_type=model_type,
            flags=train_flags,
        )

        unbatched_test_scores_data = pd.read_csv(
            os.path.join(save_dir, TEST_SCORES_FILE_NAME)
        )
        unbatched_test_scores = unbatched_test_scores_data[f"Mean {metric}"]

        # Train with batched generators
        utils.train(
            dataset_type="classification",
            metric=metric,
            save_dir=save_dir,
            model_type=model_type,
            flags=train_flags + ["--precompute_features"],
        )

        batched_test_scores_data = pd.read_csv(
            os.path.join(save_dir, TEST_SCORES_FILE_NAME)
        )
        batched_test_scores = batched_test_scores_data[f"Mean {metric}"]

        # Check results
        assert unbatched_test_scores[0] == pytest.approx(
            batched_test_scores[0], rel=DELTA * unbatched_test_scores[0]
        )
