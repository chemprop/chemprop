"""This tests the CLI functionality of predicting with a callback.
"""
import json

import pytest

from chemprop.cli.main import main

pytestmark = pytest.mark.CLI


@pytest.fixture
def data_path(data_dir):
    return str(data_dir / "test_smiles.csv")


@pytest.fixture
def model_path_classification(data_dir):
    return str(data_dir / "example_model_v2_classification_mol.pt")


@pytest.fixture
def model_path_regression(data_dir):
    return str(data_dir / "example_model_v2_regression_mol.pt")

def test_myerson_callback_classification(
    monkeypatch, data_path, model_path_classification, tmp_path
):
    args = [
        "chemprop",
        "predict",
        "-i",
        data_path,
        "--model-path",
        model_path_classification,
        "--callback",
        "myerson",
        "--output",
        str(tmp_path / "preds.csv"),
    ]

    with monkeypatch.context() as m:
        m.setattr("sys.argv", args)
        main()

    assert (tmp_path / "preds.csv").exists()
    assert (tmp_path / "preds_myerson_explanation.npz").exists()


def test_myerson_callback_json_output(monkeypatch, data_path, model_path_regression, tmp_path):
    callback_params = {"save_as_json": True}
    args = [
        "chemprop",
        "predict",
        "-i",
        data_path,
        "--model-path",
        model_path_regression,
        "--callback",
        "myerson",
        "--output",
        str(tmp_path / "preds.csv"),
        "--callback-params",
        json.dumps(callback_params),
    ]

    with monkeypatch.context() as m:
        m.setattr("sys.argv", args)
        main()

    assert (tmp_path / "preds.csv").exists()
    assert (tmp_path / "preds_myerson_explanation.json").exists()


def test_myerson_callback_regression(monkeypatch, data_path, model_path_regression, tmp_path):
    args = [
        "chemprop",
        "predict",
        "-i",
        data_path,
        "--model-path",
        model_path_regression,
        "--callback",
        "myerson",
        "--output",
        str(tmp_path / "preds.csv"),
    ]

    with monkeypatch.context() as m:
        m.setattr("sys.argv", args)
        main()

    assert (tmp_path / "preds.csv").exists()
    assert (tmp_path / "preds_myerson_explanation.npz").exists()
