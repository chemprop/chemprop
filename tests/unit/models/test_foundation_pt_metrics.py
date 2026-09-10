"""Minimal regression for foundation .pt files missing optional hparams keys."""

from pathlib import Path

import pytest
import torch

from chemprop.models import MPNN


@pytest.fixture
def model_path(data_dir):
    return data_dir / "example_model_v2_regression_mol.pt"


def test_load_from_file_tolerates_missing_metrics(tmp_path, model_path):
    # Foundation checkpoints (e.g. CheMeleon .pt) may omit hparams["metrics"].
    d = torch.load(model_path, map_location="cpu", weights_only=False)
    hparams = d["hyper_parameters"]
    assert "metrics" in hparams
    hparams.pop("metrics")
    missing_path = Path(tmp_path) / "foundation_like_missing_metrics.pt"
    torch.save(d, missing_path)

    model = MPNN.load_from_file(missing_path)
    assert isinstance(model, MPNN)
    assert model.metrics is not None
