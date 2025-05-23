"""This tests the CLI functionality of converting a v1 model to a v2 model
"""

import pytest

from chemprop.cli.main import main

pytestmark = pytest.mark.CLI


@pytest.mark.parametrize(
    "case", ["regression_mol", "quantile_regression", "mve_regression", "evidential_regression"]
)
def test_cli_convert(monkeypatch, tmp_path, data_dir, case):
    input_path = data_dir / f"example_model_v1_{case}.pt"
    output_path = tmp_path / f"example_model_v2_{case}.pt"

    args = [
        "chemprop",
        "convert",
        "-i",
        str(input_path),
        "-o",
        str(output_path),
        "--ignore-unsupported-metrics",
    ]

    with monkeypatch.context() as m:
        m.setattr("sys.argv", args)
        main()


def test_cli_convert_error(monkeypatch, tmp_path, data_dir):
    input_path = data_dir / "example_model_v1_quantile_regression.pt"
    output_path = tmp_path / "wont_be_converted.pt"

    args = ["chemprop", "convert", "-i", str(input_path), "-o", str(output_path)]

    with monkeypatch.context() as m:
        m.setattr("sys.argv", args)
        with pytest.raises(ValueError) as excinfo:
            main()
        assert "unsupported metrics" in str(excinfo.value)
