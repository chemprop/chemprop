"""This tests the CLI functionality of training and predicting a regression model on a multi-molecule.
"""

import json
import sys

import pytest

from chemprop.cli.main import main

pytestmark = pytest.mark.CLI


@pytest.fixture
def data_path(data_dir):
    return (
        str(data_dir / "regression" / "mol+mol" / "mol+mol.csv"),
        str(data_dir / "regression" / "mol+mol" / "descriptors.npz"),
        ("0", str(data_dir / "regression" / "mol+mol" / "atom_features_0.npz")),
        ("1", str(data_dir / "regression" / "mol+mol" / "atom_features_1.npz")),
        ("0", str(data_dir / "regression" / "mol+mol" / "bond_features_0.npz")),
        ("1", str(data_dir / "regression" / "mol+mol" / "atom_descriptors_1.npz")),
    )


@pytest.fixture
def model_path(data_dir):
    return str(data_dir / "example_model_v2_regression_mol+mol.pt")


def test_train_quick(monkeypatch, data_path):
    (
        input_path,
        desc_path,
        atom_feat_path_0,
        atom_feat_path_1,
        bond_feat_path_0,
        atom_desc_path_1,
    ) = data_path

    base_args = [
        "chemprop",
        "train",
        "-i",
        input_path,
        "--smiles-columns",
        "smiles",
        "solvent",
        "--epochs",
        "3",
        "--num-workers",
        "0",
        "--descriptors-path",
        desc_path,
        "--atom-features-path",
        *atom_feat_path_0,
        "--atom-features-path",
        *atom_feat_path_1,
        "--bond-features-path",
        *bond_feat_path_0,
        "--atom-descriptors-path",
        *atom_desc_path_1,
        "--show-individual-scores",
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


def test_predict_quick(monkeypatch, data_path, model_path):
    input_path, _, _, _, _, _ = data_path

    args = [
        "chemprop",
        "predict",
        "-i",
        input_path,
        "--smiles-columns",
        "smiles",
        "solvent",
        "--model-path",
        model_path,
    ]

    with monkeypatch.context() as m:
        m.setattr("sys.argv", args)
        main()


@pytest.mark.parametrize("ffn_block_index", ["0", "1"])
def test_fingerprint_quick(monkeypatch, data_path, model_path, ffn_block_index):
    input_path, _, _, _, _, _ = data_path
    args = [
        "chemprop",
        "fingerprint",
        "-i",
        input_path,
        "--smiles-columns",
        "smiles",
        "solvent",
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
        "--smiles-columns",
        "smiles",
        "solvent",
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
        "--smiles-columns",
        "smiles",
        "solvent",
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


def test_train_splits_file(monkeypatch, data_path, tmp_path):
    splits_file = str(tmp_path / "splits.json")
    splits = [
        {"train": [1, 2], "val": "3-5", "test": "6,7"},
        {"val": [1, 2], "test": "3-5", "train": "6,7"},
    ]

    with open(splits_file, "w") as f:
        json.dump(splits, f)

    input_path, *_ = data_path

    args = [
        "chemprop",
        "train",
        "-i",
        input_path,
        "--smiles-columns",
        "smiles",
        "solvent",
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


def test_train_molecule_featurizers(monkeypatch, data_path):
    input_path, descriptors_path, *_ = data_path

    args = [
        "chemprop",
        "train",
        "-i",
        input_path,
        "--smiles-columns",
        "smiles",
        "solvent",
        "--epochs",
        "3",
        "--num-workers",
        "0",
        "--descriptors-path",
        descriptors_path,
        "--molecule-featurizers",
        "morgan_count",
    ]

    with monkeypatch.context() as m:
        m.setattr("sys.argv", args)
        main()


@pytest.mark.skipif(
    sys.platform in ["win32", "darwin"], reason="Multiprocessing can hang on Windows and MacOS."
)
def test_train_multiprocess(monkeypatch, data_path):
    input_path, descriptors_path, *_ = data_path

    args = [
        "chemprop",
        "train",
        "-i",
        input_path,
        "--smiles-columns",
        "smiles",
        "solvent",
        "--epochs",
        "3",
        "--num-workers",
        "2",
        "--descriptors-path",
        descriptors_path,
        "--molecule-featurizers",
        "v1_rdkit_2d_normalized",
    ]

    with monkeypatch.context() as m:
        m.setattr("sys.argv", args)
        main()


@pytest.mark.skipif(
    sys.platform in ["win32", "darwin"], reason="Multiprocessing can hang on Windows and MacOS."
)
def test_predict_multiprocess(monkeypatch, data_path, model_path):
    input_path, _, _, _, _, _ = data_path

    args = [
        "chemprop",
        "predict",
        "-i",
        input_path,
        "--smiles-columns",
        "smiles",
        "solvent",
        "--model-path",
        model_path,
        "--num-workers",
        "2",
    ]

    with monkeypatch.context() as m:
        m.setattr("sys.argv", args)
        main()
