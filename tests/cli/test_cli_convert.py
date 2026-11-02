import pickle

import pandas as pd
import pytest

from chemprop.cli.main import main

pytestmark = [pytest.mark.CLI, pytest.mark.unsafe_load]


@pytest.fixture
def mol_data_path(data_dir):
    return str(data_dir / "regression" / "mol" / "mol.csv")


@pytest.fixture
def mol_model_file(pre_v2_4_model_path):
    return str(pre_v2_4_model_path("pre_v2_4_model_files/example_model_v2_regression_mol.pt"))


@pytest.fixture
def MAB_data_path(data_dir):
    return str(data_dir / "mol_atom_bond" / "regression.csv")


@pytest.fixture
def MAB_model_file(pre_v2_4_model_path):
    return str(pre_v2_4_model_path("pre_v2_4_model_files/mol_atom_bond/regression.pt"))


@pytest.mark.parametrize("conversion", ["latest", "v2_1_to_v2_4"])
def test_convert_then_predict_weights_only(
    monkeypatch, mol_data_path, mol_model_file, tmp_path, conversion
):
    converted_path = str(tmp_path / "converted.pt")
    args = [
        "chemprop",
        "convert",
        "-c",
        conversion,
        "-i",
        mol_model_file,
        "-o",
        converted_path,
        "--weights-only",
        "false",
    ]

    with monkeypatch.context() as m:
        m.setattr("sys.argv", args)
        main()

    for path, weights_only, output in [
        (mol_model_file, "false", "old.csv"),
        (converted_path, "true", "new.csv"),
    ]:
        args = [
            "chemprop",
            "predict",
            "-i",
            mol_data_path,
            "--model-path",
            path,
            "--weights-only",
            weights_only,
            "--output",
            str(tmp_path / output),
        ]
        with monkeypatch.context() as m:
            m.setattr("sys.argv", args)
            main()

    pd.testing.assert_frame_equal(
        pd.read_csv(tmp_path / "old.csv"), pd.read_csv(tmp_path / "new.csv")
    )


def test_convert_then_predict_weights_only_MAB(
    monkeypatch, MAB_data_path, MAB_model_file, tmp_path
):
    converted_path = str(tmp_path / "converted.pt")
    args = [
        "chemprop",
        "convert",
        "-c",
        "latest",
        "-i",
        MAB_model_file,
        "-o",
        converted_path,
        "--weights-only",
        "false",
    ]

    with monkeypatch.context() as m:
        m.setattr("sys.argv", args)
        main()

    args = [
        "chemprop",
        "predict",
        "-i",
        MAB_data_path,
        "--model-path",
        converted_path,
        "--keep-h",
        "--reorder-atoms",
        "--weights-only",
        "true",
        "--output",
        str(tmp_path / "preds.csv"),
    ]

    with monkeypatch.context() as m:
        m.setattr("sys.argv", args)
        main()


def test_convert_without_weights_only_warns(monkeypatch, mol_model_file, tmp_path):
    converted_path = tmp_path / "converted.pt"
    args = ["chemprop", "convert", "-i", mol_model_file, "-o", str(converted_path)]

    with monkeypatch.context() as m:
        m.setattr("sys.argv", args)
        with pytest.warns(FutureWarning, match="default will change to `True`"):
            main()

    assert converted_path.exists()


def test_convert_weights_only_true_raises(monkeypatch, mol_model_file, tmp_path):
    converted_path = tmp_path / "converted.pt"
    args = ["chemprop", "convert", "-i", mol_model_file, "-o", str(converted_path)]

    with monkeypatch.context() as m:
        m.setattr("sys.argv", args + ["--weights-only", "true"])
        with pytest.raises(pickle.UnpicklingError, match="--weights-only false"):
            main()

    assert not converted_path.exists()
