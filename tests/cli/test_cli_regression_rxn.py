"""This tests the CLI functionality of training and predicting a regression model on a multi-molecule.
"""

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
    args = [
        "chemprop",
        "train",
        "-i",
        input_path,
        "--reaction-columns",
        "smiles",
        "--epochs",
        "1",
        "--num-workers",
        "0",
        "--descriptors-path",
        descriptors_path,
        "--save-preds",
    ]

    with monkeypatch.context() as m:
        m.setattr("sys.argv", args)
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


@pytest.mark.parametrize("repr_type", ["FINGERPRINT", "ENCODING"])
def test_fingerprint_quick(monkeypatch, data_path, model_path, repr_type):
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
        "--repr-type",
        repr_type,
    ]

    with monkeypatch.context() as m:
        m.setattr("sys.argv", args)
        main()
