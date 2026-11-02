from argparse import Namespace
import io
import pickle
import warnings

from lightning import pytorch as pl
import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader
import torchmetrics

import chemprop
from chemprop.data import MoleculeDatapoint, MoleculeDataset, collate_batch
from chemprop.models import MPNN, MolAtomBondMPNN, MulticomponentMPNN, save_model, serialization
from chemprop.models.serialization import (
    CHEMPROP_VERSION_KEY,
    load_checkpoint,
    needs_metric_rebuild,
    rebuild_saved_transforms,
)
from chemprop.nn import (
    MAE,
    MSE,
    BondMessagePassing,
    ConstrainerFFN,
    GraphTransform,
    MABBondMessagePassing,
    MeanAggregation,
    MulticomponentMessagePassing,
    NormAggregation,
    R2Score,
    RegressionFFN,
    ScaleTransform,
    UnscaleTransform,
)
from chemprop.nn.hparams import class_path, serialize_hparams

PRE_V2_4_DIR = "pre_v2_4_model_files"
OLD_MPNN_FILES = [
    "example_model_v2_classification_dirichlet_mol.pt",
    "example_model_v2_classification_mol.pt",
    "example_model_v2_classification_mol_multiclass.pt",
    "example_model_v2_classification_mol_with_metrics.ckpt",
    "example_model_v2_multiclass_dirichlet_mol.pt",
    "example_model_v2_regression_evidential_mol.pt",
    "example_model_v2_regression_mol.ckpt",
    "example_model_v2_regression_mol.pt",
    "example_model_v2_regression_mol_multitask.pt",
    "example_model_v2_regression_mol_with_metrics.ckpt",
    "example_model_v2_regression_mve_mol.pt",
    "example_model_v2_regression_quantile_mol.pt",
    "example_model_v2_regression_rxn.ckpt",
    "example_model_v2_regression_rxn.pt",
    "example_model_v2_trained_on_cuda.pt",
]
OLD_MULTICOMPONENT_FILES = [
    "example_model_v2_regression_mol+mol+rxn_check_predictions.pt",
    "example_model_v2_regression_mol+mol.ckpt",
    "example_model_v2_regression_mol+mol.pt",
    "example_model_v2_regression_rxn+mol.pt",
]
OLD_MAB_FILES = [
    "QM_descriptors.pt",
    "atomic_regression_atom_mapped.pt",
    "classification.pt",
    "multiclass.pt",
    "regression.pt",
    "regression_constrained.pt",
    "regression_mve.pt",
    "regression_no_atom.pt",
    "regression_no_bond.pt",
    "regression_no_mol.pt",
    "regression_only_atom.pt",
    "regression_only_bond.pt",
    "regression_only_mol.pt",
    "regression_with_extras.pt",
]
OLD_FILES = (
    [(MPNN, f) for f in OLD_MPNN_FILES]
    + [(MulticomponentMPNN, f) for f in OLD_MULTICOMPONENT_FILES]
    + [(MolAtomBondMPNN, f"mol_atom_bond/{f}") for f in OLD_MAB_FILES]
)


class CustomActivation(nn.Module):
    def forward(self, X):
        return X


class CustomAggregation(MeanAggregation):
    pass


class RecordingMPNN(MPNN):
    def on_load_checkpoint(self, checkpoint):
        self.loaded_checkpoint = checkpoint


def assert_same_state_dict(model1, model2):
    sd1, sd2 = model1.state_dict(), model2.state_dict()
    assert sd1.keys() == sd2.keys()
    for key in sd1:
        assert torch.equal(sd1[key], sd2[key]), key


def assert_weights_only_safe(path):
    assert torch.serialization.get_unsafe_globals_in_checkpoint(path) == []
    torch.load(path, weights_only=True)


@pytest.fixture
def mpnn():
    graph_transform = GraphTransform(
        V_transform=nn.Identity(), E_transform=ScaleTransform(mean=[0.0, 1.0], scale=[2.0, 3.0])
    )
    mp = BondMessagePassing(
        d_vd=3,
        activation=nn.LeakyReLU(0.2),
        graph_transform=graph_transform,
        V_d_transform=ScaleTransform(mean=[4.0, 5.0], scale=[6.0, 7.0], pad=1),
    )
    ffn = RegressionFFN(
        n_tasks=2,
        input_dim=mp.output_dim + 2,
        criterion=MSE(task_weights=[0.5, 2.0]),
        output_transform=UnscaleTransform(mean=[8.0, 9.0], scale=[10.0, 11.0]),
    )
    X_d_transform = ScaleTransform(mean=[13.0, 14.0], scale=[15.0, 16.0])

    return MPNN(
        mp, NormAggregation(norm=50), ffn, metrics=[MAE(), R2Score()], X_d_transform=X_d_transform
    )


@pytest.fixture
def multicomponent_mpnn():
    blocks = [BondMessagePassing(), BondMessagePassing(d_h=200)]
    mp = MulticomponentMessagePassing(blocks, n_components=2)
    ffn = RegressionFFN(input_dim=mp.output_dim)

    return MulticomponentMPNN(mp, MeanAggregation(), ffn, metrics=[MAE()])


@pytest.fixture
def mab_mpnn():
    mp = MABBondMessagePassing(
        d_ed=2, E_d_transform=ScaleTransform(mean=[1.0, 2.0], scale=[3.0, 4.0])
    )

    return MolAtomBondMPNN(
        message_passing=mp,
        agg=MeanAggregation(),
        mol_predictor=RegressionFFN(),
        atom_predictor=RegressionFFN(),
        bond_predictor=RegressionFFN(input_dim=mp.output_dims[1] * 2),
        atom_constrainer=ConstrainerFFN(),
        bond_constrainer=ConstrainerFFN(fp_dim=mp.output_dims[1] * 2),
        metrics=[MAE()],
    )


@pytest.fixture
def model_path(tmp_path, mpnn):
    path = tmp_path / "model.pt"
    save_model(path, mpnn)

    return path


@pytest.fixture
def unsafe_path(tmp_path):
    """a model file with an arbitrary object, like those saved before Chemprop v2.4"""
    path = tmp_path / "unsafe.pt"
    torch.save({"hyper_parameters": {"args": Namespace()}, "state_dict": {}}, path)

    return path


@pytest.fixture
def loader(smis):
    data = [MoleculeDatapoint.from_smi(smi, np.array([0.0])) for smi in smis[:16]]

    return DataLoader(MoleculeDataset(data), 8, collate_fn=collate_batch)


@pytest.fixture
def unconvertible_checkpoint_path(tmp_path, data_dir):
    checkpoint = torch.load(data_dir / "example_model_v2_regression_mol.ckpt", weights_only=True)
    checkpoint["callbacks"]["custom"] = CustomActivation()
    path = tmp_path / "model.ckpt"
    torch.save(checkpoint, path)

    return path


# ======== saving and loading model files from Chemprop v2.4+ ========


@pytest.mark.parametrize("model", ["mpnn", "multicomponent_mpnn", "mab_mpnn"])
def test_roundtrip(tmp_path, request, model):
    model = request.getfixturevalue(model)
    path = tmp_path / "model.pt"
    save_model(path, model)
    assert_weights_only_safe(path)
    assert torch.load(path, weights_only=True)[CHEMPROP_VERSION_KEY] == chemprop.__version__

    loaded = type(model).load_from_file(path, weights_only=True)

    assert_same_state_dict(model, loaded)


def test_roundtrip_hparams_outside_state_dict(model_path):
    loaded = MPNN.load_from_file(model_path, weights_only=True)

    assert isinstance(loaded.message_passing.tau, nn.LeakyReLU)
    assert loaded.message_passing.tau.negative_slope == 0.2
    assert loaded.agg.norm == 50
    assert [type(m) for m in loaded.metrics] == [MAE, R2Score, MSE]


def test_lightning_checkpoint_is_weights_only_safe(tmp_path, loader):
    model = MPNN(
        BondMessagePassing(),
        MeanAggregation(),
        RegressionFFN(output_transform=UnscaleTransform(mean=[1.0], scale=[2.0])),
        metrics=[MAE()],
    )
    trainer = pl.Trainer(
        logger=False,
        max_epochs=1,
        accelerator="cpu",
        enable_progress_bar=False,
        enable_model_summary=False,
        default_root_dir=tmp_path,
    )
    trainer.fit(model, loader, loader)
    ckpt_path = trainer.checkpoint_callback.best_model_path

    assert_weights_only_safe(ckpt_path)
    assert torch.load(ckpt_path, weights_only=True)[CHEMPROP_VERSION_KEY] == chemprop.__version__
    assert_same_state_dict(model, MPNN.load_from_checkpoint(ckpt_path, weights_only=True))


def test_load_from_checkpoint_overrides(model_path):
    agg = NormAggregation(norm=50)

    loaded = MPNN.load_from_checkpoint(model_path, weights_only=True, init_lr=0.5, agg=agg)

    assert loaded.init_lr == 0.5
    assert loaded.agg is agg


def test_custom_classes(tmp_path):
    model = MPNN(
        BondMessagePassing(activation=CustomActivation()), CustomAggregation(), RegressionFFN()
    )
    path = tmp_path / "model.pt"
    save_model(path, model)

    # the custom aggregation has hparams, so it is saved as a class path, but the custom
    # activation can't be converted, so the object itself is saved
    assert torch.serialization.get_unsafe_globals_in_checkpoint(path) == [
        class_path(CustomActivation)
    ]
    with pytest.raises(pickle.UnpicklingError):
        MPNN.load_from_file(path, weights_only=True)

    loaded = MPNN.load_from_file(path, weights_only=False)
    assert isinstance(loaded.agg, CustomAggregation)
    assert isinstance(loaded.message_passing.tau, CustomActivation)


# ======== passing the checkpoint to Lightning ========


@pytest.mark.parametrize(
    "lightning_supports_weights_only",
    [
        pytest.param(
            True,
            marks=pytest.mark.skipif(
                not serialization.LIGHTNING_SUPPORTS_WEIGHTS_ONLY,
                reason="Lightning doesn't support passing `weights_only`",
            ),
        ),
        False,
    ],
)
def test_on_load_checkpoint_gets_training_state(
    monkeypatch, data_dir, lightning_supports_weights_only
):
    monkeypatch.setattr(
        serialization, "LIGHTNING_SUPPORTS_WEIGHTS_ONLY", lightning_supports_weights_only
    )
    path = data_dir / "example_model_v2_regression_mol.ckpt"

    model = RecordingMPNN.load_from_checkpoint(path, weights_only=True)

    checkpoint = torch.load(path, weights_only=True)
    assert model.loaded_checkpoint.keys() >= checkpoint.keys()
    assert model.loaded_checkpoint["epoch"] == checkpoint["epoch"]


@pytest.mark.skipif(
    not serialization.LIGHTNING_SUPPORTS_WEIGHTS_ONLY,
    reason="Lightning doesn't support passing `weights_only`",
)
def test_on_load_checkpoint_with_unconvertible_objects(unconvertible_checkpoint_path):
    model = RecordingMPNN.load_from_checkpoint(unconvertible_checkpoint_path, weights_only=False)

    assert isinstance(model.loaded_checkpoint["callbacks"]["custom"], CustomActivation)


@pytest.mark.skipif(
    torch.__version__ < "2.6", reason="PyTorch defaults to `weights_only=False` before v2.6"
)
def test_old_lightning_with_unconvertible_objects(monkeypatch, unconvertible_checkpoint_path):
    monkeypatch.setattr(serialization, "LIGHTNING_SUPPORTS_WEIGHTS_ONLY", False)

    with pytest.raises(pickle.UnpicklingError, match="lightning>=2.6"):
        MPNN.load_from_checkpoint(unconvertible_checkpoint_path, weights_only=False)


# ======== `weights_only` handling ========


def test_weights_only_default_warns(model_path):
    for load in (MPNN.load_from_file, load_checkpoint):
        with pytest.warns(FutureWarning, match="default will change to `True`"):
            load(model_path)
        with warnings.catch_warnings():
            warnings.simplefilter("error", FutureWarning)
            load(model_path, weights_only=True)


def test_load_unsafe_file_with_weights_only_raises(unsafe_path):
    with pytest.raises(pickle.UnpicklingError, match="chemprop convert"):
        MPNN.load_from_file(unsafe_path, weights_only=True)


# ======== loading pre-v2.4 model files ========


def test_needs_metric_rebuild():
    assert not needs_metric_rebuild(MSE())
    assert not needs_metric_rebuild(torchmetrics.MeanSquaredError())
    assert not needs_metric_rebuild(None)


def test_rebuild_saved_transforms():
    """transforms unpickled from pre-v2.4 model files don't have hparams, so they are rebuilt"""
    X_d = ScaleTransform([1.0, 2.0], [3.0, 4.0])
    V_d = ScaleTransform([5.0], [6.0], pad=1)
    output = UnscaleTransform([7.0], [8.0])
    graph = GraphTransform(ScaleTransform([1.0], [2.0]), ScaleTransform([3.0], [4.0]))
    for transform in (X_d, V_d, output, graph, graph.V_transform, graph.E_transform):
        del transform.hparams
    new_output = UnscaleTransform([9.0], [10.0])
    hparams = {
        "X_d_transform": X_d,
        "message_passing": {"graph_transform": graph, "blocks": [{"V_d_transform": V_d}]},
        "predictor": {"output_transform": output},
        "mol_predictor": {"output_transform": new_output},
    }

    rebuild_saved_transforms(hparams)

    torch.testing.assert_close(hparams["X_d_transform"].scale, X_d.scale)
    torch.testing.assert_close(
        hparams["message_passing"]["blocks"][0]["V_d_transform"].mean, V_d.mean
    )
    assert type(hparams["predictor"]["output_transform"]) is UnscaleTransform
    assert hparams["mol_predictor"]["output_transform"] is new_output
    buffer = io.BytesIO()
    torch.save(serialize_hparams(hparams), buffer)
    buffer.seek(0)
    torch.load(buffer, weights_only=True)


@pytest.mark.unsafe_load
@pytest.mark.parametrize("cls, filename", OLD_FILES, ids=[f for _, f in OLD_FILES])
def test_load_old_files(pre_v2_4_model_path, cls, filename):
    path = pre_v2_4_model_path(f"{PRE_V2_4_DIR}/{filename}")

    cls.load_from_file(path, map_location="cpu", weights_only=False)
