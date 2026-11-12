"""This tests the CLI functionality of training and predicting a regression model on a multi-molecule.
"""

from configargparse import ArgumentError
import pytest

from chemprop.cli.main import main

pytestmark = pytest.mark.CLI


@pytest.fixture
def data_path(data_dir):
    return str(data_dir / "regression" / "rxn" / "rxn.csv"), str(
        data_dir / "regression" / "rxn" / "descriptors.npz"
    )


@pytest.fixture
def model_path(data_dir):
    return str(data_dir / "example_model_v2_regression_rxn.pt")


def test_train_quick(monkeypatch, data_path):
    input_path, descriptors_path = data_path

    base_args = [
        "chemprop",
        "train",
        "-i",
        input_path,
        "--reaction-columns",
        "smiles",
        "--epochs",
        "3",
        "--num-workers",
        "0",
        "--descriptors-path",
        descriptors_path,
        "--show-individual-scores",
    ]

    task_types = ["", "regression-mve", "regression-evidential", "regression-quantile"]

    for task_type in task_types:
        args = base_args.copy()

        if task_type:
            args.extend(["--task-type", task_type])

        if task_type == "regression-evidential":
            args += ["--evidential-regularization", "0.2"]

        with monkeypatch.context() as m:
            m.setattr("sys.argv", args)
            main()


def test_train_quick_from_foundation(monkeypatch, data_path):
    input_path, _ = data_path

    args = [
        "chemprop",
        "train",
        "-i",
        input_path,
        "--reaction-columns",
        "smiles",
        "--rxn-mode",
        "REAC_DIFF",
        "--keep-h",
        "--epochs",
        "1",
        "--warmup-epochs",
        "0",
        "--init-lr",
        "1e-4",
        "--max-lr",
        "1e-4",
        "--final-lr",
        "1e-4",
        "--num-workers",
        "0",
        "--show-individual-scores",
        "--from-foundation",
        "chemeleon-rxn",
    ]

    with monkeypatch.context() as m:
        m.setattr("sys.argv", args)
        main()


@pytest.mark.parametrize(
    "bad_args", [["--rxn-mode", "REAC_DIFF_BALANCE", "--keep-h"], ["--rxn-mode", "REAC_DIFF"]]
)
def test_train_from_foundation_featurization_mismatch(monkeypatch, data_path, bad_args):
    input_path, _ = data_path

    args = [
        "chemprop",
        "train",
        "-i",
        input_path,
        "--reaction-columns",
        "smiles",
        *bad_args,
        "--from-foundation",
        "chemeleon-rxn",
    ]

    with monkeypatch.context() as m:
        m.setattr("sys.argv", args)
        with pytest.raises(ArgumentError, match="CheMeleon-Rxn must be used with"):
            main()


def test_predict_quick(monkeypatch, data_path, model_path):
    input_path, _ = data_path
    args = [
        "chemprop",
        "predict",
        "-i",
        input_path,
        "--reaction-columns",
        "smiles",
        "--model-path",
        model_path,
    ]

    with monkeypatch.context() as m:
        m.setattr("sys.argv", args)
        main()


@pytest.mark.parametrize("ffn_block_index", ["0", "1"])
def test_fingerprint_quick(monkeypatch, data_path, model_path, ffn_block_index):
    input_path, _ = data_path
    args = [
        "chemprop",
        "fingerprint",
        "-i",
        input_path,
        "--reaction-columns",
        "smiles",
        "--model-path",
        model_path,
        "--ffn-block-index",
        ffn_block_index,
    ]

    with monkeypatch.context() as m:
        m.setattr("sys.argv", args)
        main()
