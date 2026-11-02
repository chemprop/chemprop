from argparse import Namespace
import csv
import os
import pickle
import subprocess
import sys

from lightning import pytorch as pl
import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

import chemprop
from chemprop.data import collate_batch
from chemprop.data.dataloader import build_dataloader
from chemprop.data.datapoints import MoleculeDatapoint
from chemprop.data.datasets import MoleculeDataset
from chemprop.featurizers.atom import MultiHotAtomFeaturizer
from chemprop.featurizers.molgraph.molecule import SimpleMoleculeMolGraphFeaturizer
from chemprop.models import save_model
from chemprop.models.model import MPNN
from chemprop.models.serialization import CHEMPROP_VERSION_KEY
from chemprop.nn import BondMessagePassing, MeanAggregation, RegressionFFN
from chemprop.utils.convert_to_latest import convert_model_file_to_latest
from chemprop.utils.v1_to_v2 import convert_model_file_v1_to_v2
from chemprop.utils.v2_0_to_v2_1 import convert_model_file_v2_0_to_v2_1
from chemprop.utils.v2_1_to_v2_4 import convert_model_file_v2_1_to_v2_4

CONVERT_FILE_FUNCTIONS = [convert_model_file_v2_1_to_v2_4, convert_model_file_to_latest]


def assert_weights_only_safe(path):
    assert torch.serialization.get_unsafe_globals_in_checkpoint(path) == []
    torch.load(path, weights_only=True)


@pytest.fixture
def example_model_v1_path(pre_v2_4_model_path):
    return pre_v2_4_model_path("v1_model_files/example_model_v1_regression_mol.pt")


@pytest.fixture
def example_model_v1_4_path(pre_v2_4_model_path):
    return pre_v2_4_model_path("v1_model_files/example_model_v1_4.pt")


@pytest.fixture
def example_model_v1_prediction(data_dir):
    path = data_dir / "example_model_v1_regression_mol_prediction.csv"

    with open(path) as fid:
        reader = csv.reader(fid)
        next(reader)
        smis, ys = zip(*[(smi, float(score)) for smi, score in reader])

    featurizer = SimpleMoleculeMolGraphFeaturizer(atom_featurizer=MultiHotAtomFeaturizer.v1())

    ys = np.array(ys).reshape(-1, 1)
    test_data = [MoleculeDatapoint.from_smi(smi, None) for smi in smis]
    test_dset = MoleculeDataset(test_data, featurizer)

    test_loader = build_dataloader(test_dset, shuffle=False)
    return ys, test_loader


@pytest.fixture
def model_path(tmp_path):
    path = tmp_path / "model.pt"
    save_model(path, MPNN(BondMessagePassing(), MeanAggregation(), RegressionFFN()))

    return path


@pytest.fixture
def unsafe_path(tmp_path):
    """a model file with an arbitrary object, like those saved before Chemprop v2.4"""
    path = tmp_path / "unsafe.pt"
    torch.save({"hyper_parameters": {"args": Namespace()}, "state_dict": {}}, path)

    return path


@pytest.fixture
def loader(smis):
    data = [MoleculeDatapoint.from_smi(smi) for smi in smis[:16]]

    return DataLoader(MoleculeDataset(data), 8, collate_fn=collate_batch)


@pytest.mark.unsafe_load
def test_converter(tmp_path, example_model_v1_path, example_model_v1_prediction):
    directory = tmp_path / "test_converter"
    directory.mkdir()
    model_v2_save_path = directory / "example_model_v2_regression_mol.pt"

    convert_model_file_v1_to_v2(example_model_v1_path, model_v2_save_path, weights_only=False)
    assert model_v2_save_path.exists()

    mpnn = MPNN.load_from_checkpoint(model_v2_save_path, weights_only=False)

    ys_v1, test_loader = example_model_v1_prediction

    trainer = pl.Trainer(accelerator="cpu", logger=None, enable_progress_bar=False)
    predss = trainer.predict(mpnn, test_loader)
    ys_v2 = np.vstack(predss)
    assert np.allclose(ys_v2, ys_v1, atol=1e-6)


@pytest.mark.unsafe_load
def test_converter_v1_4(tmp_path, example_model_v1_4_path):
    directory = tmp_path / "test_converter"
    directory.mkdir()
    model_v2_save_path = directory / "converted_v1_4.pt"

    convert_model_file_v1_to_v2(example_model_v1_4_path, model_v2_save_path, weights_only=False)
    assert model_v2_save_path.exists()


@pytest.mark.unsafe_load
def test_convert_v1_to_latest(tmp_path, example_model_v1_path, example_model_v1_prediction):
    save_path = tmp_path / "example_model_latest.pt"

    convert_model_file_to_latest(example_model_v1_path, save_path, weights_only=False)
    assert torch.serialization.get_unsafe_globals_in_checkpoint(save_path) == []

    mpnn = MPNN.load_from_checkpoint(save_path, weights_only=True)

    ys_v1, test_loader = example_model_v1_prediction

    trainer = pl.Trainer(accelerator="cpu", logger=None, enable_progress_bar=False)
    predss = trainer.predict(mpnn, test_loader)
    ys_v2 = np.vstack(predss)
    assert np.allclose(ys_v2, ys_v1, atol=1e-6)


@pytest.mark.unsafe_load
def test_convert_v1_4_to_latest(tmp_path, example_model_v1_4_path):
    save_path = tmp_path / "converted_v1_4_latest.pt"

    convert_model_file_to_latest(example_model_v1_4_path, save_path, weights_only=False)

    assert torch.serialization.get_unsafe_globals_in_checkpoint(save_path) == []


@pytest.mark.unsafe_load
@pytest.mark.parametrize("convert", CONVERT_FILE_FUNCTIONS)
@pytest.mark.parametrize(
    "filename",
    ["example_model_v2_regression_mol_with_metrics.ckpt", "example_model_v2_trained_on_cuda.pt"],
)
def test_convert_to_v2_4(tmp_path, pre_v2_4_model_path, loader, convert, filename):
    path = pre_v2_4_model_path(f"pre_v2_4_model_files/{filename}")
    output = tmp_path / filename
    convert(path, output, weights_only=False)
    assert_weights_only_safe(output)

    # lightning checkpoints keep the training state
    old_keys = torch.load(path, map_location="cpu", weights_only=False).keys()
    converted = torch.load(output, weights_only=True)
    assert converted.keys() == old_keys | {CHEMPROP_VERSION_KEY}
    assert converted[CHEMPROP_VERSION_KEY] == "2.4.0"

    old = MPNN.load_from_file(path, map_location="cpu", weights_only=False)
    new = MPNN.load_from_file(output, weights_only=True)
    trainer = pl.Trainer(logger=False, accelerator="cpu", enable_progress_bar=False)
    old_preds = torch.cat(trainer.predict(old, loader))
    new_preds = torch.cat(trainer.predict(new, loader))
    assert torch.equal(old_preds, new_preds)


@pytest.mark.parametrize("convert", CONVERT_FILE_FUNCTIONS)
def test_convert_v2_4_file(tmp_path, model_path, convert):
    output = tmp_path / "converted.pt"
    with pytest.warns(FutureWarning, match="default will change to `True`"):
        convert(model_path, output)

    assert_weights_only_safe(output)
    assert torch.load(output, weights_only=True)[CHEMPROP_VERSION_KEY] == chemprop.__version__
    MPNN.load_from_file(output, weights_only=True)


@pytest.mark.parametrize(
    "convert",
    [convert_model_file_v1_to_v2, convert_model_file_v2_0_to_v2_1, *CONVERT_FILE_FUNCTIONS],
)
def test_convert_unsafe_file_with_weights_only_raises(tmp_path, unsafe_path, convert):
    output = tmp_path / "converted.pt"

    # the conversions that need a custom unpickler raise a `RuntimeError` with `weights_only=True`
    with pytest.raises((pickle.UnpicklingError, RuntimeError)):
        convert(unsafe_path, output, weights_only=True)

    assert not output.exists()


@pytest.mark.parametrize("env_var, warns", [(None, True), ("1", False)])
def test_conversion_script_env_var(tmp_path, model_path, env_var, warns):
    """the conversion scripts can't be passed `weights_only`, so they mention the env var"""
    env = {k: v for k, v in os.environ.items() if k != "CHEMPROP_ALLOW_UNSAFE_LOAD"}
    if env_var is not None:
        env["CHEMPROP_ALLOW_UNSAFE_LOAD"] = env_var

    result = subprocess.run(
        [sys.executable, "-m", "chemprop.utils.v2_1_to_v2_4", model_path, tmp_path / "out.pt"],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )

    assert ("CHEMPROP_ALLOW_UNSAFE_LOAD=1" in result.stderr) == warns
