import io
import pickle
import warnings

from lightning import pytorch as pl
import numpy as np
import pytest
import torch
from torch import nn
import torchmetrics
from torch.utils.data import DataLoader

from chemprop.data import MoleculeDatapoint, MoleculeDataset, collate_batch
from chemprop.models import MPNN, MolAtomBondMPNN, MulticomponentMPNN, save_model
from chemprop.models import serialization
from chemprop.models.serialization import (
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
from chemprop.utils.v2_1_to_v2_4 import (
    convert_model_file_to_latest,
    convert_model_file_v2_1_to_v2_4,
)

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


class CustomActivation(nn.Module):
    def forward(self, X):
        return X


class CustomAggregation(MeanAggregation):
    pass


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


@pytest.mark.parametrize("weights_only", [True, False])
def test_mpnn_roundtrip(tmp_path, mpnn, weights_only):
    path = tmp_path / "model.pt"
    save_model(path, mpnn, ["a", "b"])
    assert_weights_only_safe(path)

    loaded = MPNN.load_from_file(path, weights_only=weights_only)

    assert_same_state_dict(mpnn, loaded)
    assert isinstance(loaded.message_passing.tau, nn.LeakyReLU)
    assert loaded.message_passing.tau.negative_slope == 0.2
    assert loaded.agg.norm == 50
    assert [type(m) for m in loaded.metrics] == [MAE, R2Score, MSE]
    assert torch.equal(loaded.predictor.criterion.task_weights, torch.tensor([[0.5, 2.0]]))
    assert torch.equal(loaded.message_passing.V_d_transform.mean, torch.tensor([[0.0, 4.0, 5.0]]))
    assert torch.equal(
        loaded.message_passing.graph_transform.E_transform.scale, torch.tensor([[2.0, 3.0]])
    )
    assert torch.equal(loaded.predictor.output_transform.mean, torch.tensor([[8.0, 9.0]]))
    assert torch.equal(loaded.X_d_transform.scale, torch.tensor([[15.0, 16.0]]))


@pytest.mark.parametrize("weights_only", [True, False])
def test_multicomponent_roundtrip(tmp_path, multicomponent_mpnn, weights_only):
    path = tmp_path / "model.pt"
    save_model(path, multicomponent_mpnn)
    assert_weights_only_safe(path)

    loaded = MulticomponentMPNN.load_from_file(path, weights_only=weights_only)

    assert_same_state_dict(multicomponent_mpnn, loaded)
    assert [block.output_dim for block in loaded.message_passing.blocks] == [300, 200]


@pytest.mark.parametrize("weights_only", [True, False])
def test_mab_roundtrip(tmp_path, mab_mpnn, weights_only):
    path = tmp_path / "model.pt"
    save_model(path, mab_mpnn)
    assert_weights_only_safe(path)

    loaded = MolAtomBondMPNN.load_from_file(path, weights_only=weights_only)

    assert_same_state_dict(mab_mpnn, loaded)
    assert torch.equal(loaded.message_passing.E_d_transform.mean, torch.tensor([[1.0, 2.0]]))
    assert isinstance(loaded.bond_constrainer, ConstrainerFFN)


def test_lightning_checkpoint_is_weights_only_safe(tmp_path, smis):
    data = [MoleculeDatapoint.from_smi(smi, np.array([0.0])) for smi in smis[:16]]
    loader = DataLoader(MoleculeDataset(data), 8, collate_fn=collate_batch)
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
    loaded = MPNN.load_from_checkpoint(ckpt_path, weights_only=True)
    assert_same_state_dict(model, loaded)


@pytest.mark.unsafe_load
@pytest.mark.parametrize(
    "filename",
    OLD_MPNN_FILES + OLD_MULTICOMPONENT_FILES + [f"mol_atom_bond/{f}" for f in OLD_MAB_FILES],
)
def test_new_model_files_match_pre_v2_4(data_dir, pre_v2_4_model_path, filename):
    """the model files in `tests/data` were converted from the pre-v2.4 model files"""
    pre_v2_4_path = pre_v2_4_model_path(f"{PRE_V2_4_DIR}/{filename}")
    new_path = data_dir / filename.replace("mol_atom_bond/", "mol_atom_bond/example_models/")
    assert_weights_only_safe(new_path)

    pre_v2_4 = torch.load(pre_v2_4_path, map_location="cpu", weights_only=False)["state_dict"]
    new = torch.load(new_path, weights_only=True)["state_dict"]
    assert pre_v2_4.keys() == new.keys()
    for key in pre_v2_4:
        assert torch.equal(pre_v2_4[key], new[key]), key


@pytest.mark.unsafe_load
@pytest.mark.parametrize("filename", OLD_MPNN_FILES + ["example_model_v2_trained_on_cuda.pt"])
def test_old_mpnn_files(pre_v2_4_model_path, filename):
    path = pre_v2_4_model_path(f"{PRE_V2_4_DIR}/{filename}")
    from_checkpoint = MPNN.load_from_checkpoint(path, map_location="cpu", weights_only=False)
    from_file = MPNN.load_from_file(path, map_location="cpu", weights_only=False)
    assert_same_state_dict(from_checkpoint, from_file)

    with pytest.raises(pickle.UnpicklingError, match="chemprop convert"):
        MPNN.load_from_file(path, map_location="cpu", weights_only=True)


@pytest.mark.unsafe_load
@pytest.mark.parametrize("filename", OLD_MULTICOMPONENT_FILES)
def test_old_multicomponent_files(pre_v2_4_model_path, filename):
    path = pre_v2_4_model_path(f"{PRE_V2_4_DIR}/{filename}")
    MulticomponentMPNN.load_from_file(path, weights_only=False)

    with pytest.raises(pickle.UnpicklingError):
        MulticomponentMPNN.load_from_file(path, weights_only=True)


@pytest.mark.unsafe_load
@pytest.mark.parametrize("filename", OLD_MAB_FILES)
def test_old_mab_files(pre_v2_4_model_path, filename):
    path = pre_v2_4_model_path(f"{PRE_V2_4_DIR}/mol_atom_bond/{filename}")
    MolAtomBondMPNN.load_from_file(path, weights_only=False)

    with pytest.raises(pickle.UnpicklingError):
        MolAtomBondMPNN.load_from_file(path, weights_only=True)


@pytest.mark.unsafe_load
def test_resaved_old_file_matches_predictions(tmp_path, pre_v2_4_model_path, smis):
    old = MPNN.load_from_file(
        pre_v2_4_model_path(f"{PRE_V2_4_DIR}/example_model_v2_regression_mol.pt"),
        map_location="cpu",
        weights_only=False,
    )
    save_model(tmp_path / "model.pt", old)
    new = MPNN.load_from_file(tmp_path / "model.pt", weights_only=True)

    data = [MoleculeDatapoint.from_smi(smi) for smi in smis]
    loader = DataLoader(MoleculeDataset(data), 32, collate_fn=collate_batch)
    trainer = pl.Trainer(logger=False, accelerator="cpu", enable_progress_bar=False)
    old_preds = torch.cat(trainer.predict(old, loader))
    new_preds = torch.cat(trainer.predict(new, loader))

    assert torch.equal(old_preds, new_preds)


@pytest.mark.unsafe_load
def test_old_metrics_are_rebuilt(pre_v2_4_model_path):
    """unpickled metrics don't have hparams, so they are rebuilt when loaded"""
    path = pre_v2_4_model_path(f"{PRE_V2_4_DIR}/example_model_v2_regression_mol_with_metrics.ckpt")
    checkpoint = load_checkpoint(path, map_location="cpu", weights_only=False)
    metrics = checkpoint["hyper_parameters"]["metrics"]
    assert metrics and all(needs_metric_rebuild(metric) for metric in metrics)

    model = MPNN.load_from_checkpoint(path, map_location="cpu", weights_only=False)
    assert all("cls" in metric.hparams for metric in model.metrics)


def test_needs_metric_rebuild():
    assert not needs_metric_rebuild(MSE())
    # metrics that aren't Chemprop's might lose arguments when rebuilt, so they aren't
    assert not needs_metric_rebuild(torchmetrics.MeanSquaredError())
    assert not needs_metric_rebuild(None)


def test_rebuild_saved_transforms():
    """transforms unpickled from pre-v2.4 model files don't have hparams, so they are rebuilt"""
    X_d = ScaleTransform([1.0, 2.0], [3.0, 4.0])
    V_d = ScaleTransform([5.0], [6.0], pad=1)
    output = UnscaleTransform([7.0], [8.0])
    graph = GraphTransform(ScaleTransform([1.0], [2.0]), ScaleTransform([3.0], [4.0]))
    for transform in (X_d, V_d, output, graph, graph.V_transform, graph.E_transform):
        # unpickling restores the old attributes without calling `__init__`
        del transform.hparams
    new_output = UnscaleTransform([9.0], [10.0])
    hparams = {
        "X_d_transform": X_d,
        "message_passing": {"graph_transform": graph, "blocks": [{"V_d_transform": V_d}]},
        "predictor": {"output_transform": output},
        "mol_predictor": {"output_transform": new_output},
    }

    rebuild_saved_transforms(hparams)

    rebuilt = {
        X_d: hparams["X_d_transform"],
        V_d: hparams["message_passing"]["blocks"][0]["V_d_transform"],
        output: hparams["predictor"]["output_transform"],
        graph.V_transform: hparams["message_passing"]["graph_transform"].V_transform,
        graph.E_transform: hparams["message_passing"]["graph_transform"].E_transform,
    }
    for old, new in rebuilt.items():
        assert type(new) is type(old) and "hparams" in vars(new)
        torch.testing.assert_close(new.mean, old.mean)
        torch.testing.assert_close(new.scale, old.scale)
    new_graph = hparams["message_passing"]["graph_transform"]
    assert type(new_graph) is GraphTransform and "hparams" in vars(new_graph)
    assert hparams["mol_predictor"]["output_transform"] is new_output

    buffer = io.BytesIO()
    torch.save(serialize_hparams(hparams), buffer)
    buffer.seek(0)
    torch.load(buffer, weights_only=True)


@pytest.mark.unsafe_load
@pytest.mark.parametrize("convert", [convert_model_file_v2_1_to_v2_4, convert_model_file_to_latest])
@pytest.mark.parametrize(
    "filename",
    ["example_model_v2_regression_mol_with_metrics.ckpt", "example_model_v2_trained_on_cuda.pt"],
)
def test_convert_to_v2_4(tmp_path, pre_v2_4_model_path, convert, filename):
    path = pre_v2_4_model_path(f"{PRE_V2_4_DIR}/{filename}")
    output = tmp_path / filename
    convert(path, output)
    assert_weights_only_safe(output)

    old = MPNN.load_from_file(path, map_location="cpu", weights_only=False)
    new = MPNN.load_from_file(output, weights_only=True)
    assert_same_state_dict(old, new)

    # lightning checkpoints keep the training state
    old_keys = torch.load(path, map_location="cpu", weights_only=False).keys()
    assert torch.load(output, weights_only=True).keys() == old_keys


def test_weights_only_default_warns(tmp_path, mpnn):
    path = tmp_path / "model.pt"
    save_model(path, mpnn)

    with pytest.warns(FutureWarning, match="default will change to `True`"):
        MPNN.load_from_file(path)
    with pytest.warns(FutureWarning, match="default will change to `True`"):
        load_checkpoint(path)

    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        MPNN.load_from_file(path, weights_only=True)
        MPNN.load_from_file(path, weights_only=False)
        load_checkpoint(path, weights_only=True)


class RecordingMPNN(MPNN):
    def on_load_checkpoint(self, checkpoint):
        self.loaded_checkpoint = checkpoint


@pytest.mark.parametrize(
    "path, weights_only",
    [
        ("example_model_v2_regression_mol.ckpt", True),
        pytest.param(
            f"{PRE_V2_4_DIR}/example_model_v2_regression_mol.ckpt",
            False,
            marks=pytest.mark.unsafe_load,
        ),
    ],
)
def test_on_load_checkpoint_gets_training_state(data_dir, pre_v2_4_model_path, path, weights_only):
    path = data_dir / path if weights_only else pre_v2_4_model_path(path)
    model = RecordingMPNN.load_from_checkpoint(path, weights_only=weights_only)

    expected = torch.load(path, map_location="cpu", weights_only=False)
    for key in ["optimizer_states", "lr_schedulers", "callbacks", "loops", "epoch"]:
        assert key in model.loaded_checkpoint
    assert model.loaded_checkpoint["epoch"] == expected["epoch"]
    assert isinstance(model.loaded_checkpoint["hyper_parameters"]["predictor"], RegressionFFN)


@pytest.fixture
def unconvertible_checkpoint_path(tmp_path, data_dir):
    checkpoint = torch.load(data_dir / "example_model_v2_regression_mol.ckpt", weights_only=True)
    checkpoint["callbacks"]["custom"] = CustomActivation()
    path = tmp_path / "model.ckpt"
    torch.save(checkpoint, path)

    return path


@pytest.mark.skipif(
    not serialization.LIGHTNING_SUPPORTS_WEIGHTS_ONLY,
    reason="Lightning doesn't support passing `weights_only`",
)
def test_on_load_checkpoint_with_unconvertible_objects(data_dir, unconvertible_checkpoint_path):
    model = RecordingMPNN.load_from_checkpoint(unconvertible_checkpoint_path, weights_only=False)

    assert isinstance(model.loaded_checkpoint["callbacks"]["custom"], CustomActivation)
    assert "optimizer_states" in model.loaded_checkpoint
    expected = MPNN.load_from_checkpoint(
        data_dir / "example_model_v2_regression_mol.ckpt", weights_only=True
    )
    assert_same_state_dict(model, expected)


@pytest.mark.skipif(
    torch.__version__ < "2.6", reason="PyTorch defaults to `weights_only=False` before v2.6"
)
def test_old_lightning_with_unconvertible_objects(monkeypatch, unconvertible_checkpoint_path):
    monkeypatch.setattr(serialization, "LIGHTNING_SUPPORTS_WEIGHTS_ONLY", False)

    with pytest.raises(pickle.UnpicklingError, match="lightning>=2.6"):
        MPNN.load_from_checkpoint(unconvertible_checkpoint_path, weights_only=False)


def test_old_lightning_with_weights_only(monkeypatch, data_dir):
    monkeypatch.setattr(serialization, "LIGHTNING_SUPPORTS_WEIGHTS_ONLY", False)
    path = data_dir / "example_model_v2_regression_mol.ckpt"

    model = RecordingMPNN.load_from_checkpoint(path, weights_only=True)

    assert "optimizer_states" in model.loaded_checkpoint
    assert_same_state_dict(model, MPNN.load_from_checkpoint(path, weights_only=True))


def test_load_from_checkpoint_overrides(tmp_path, mpnn):
    path = tmp_path / "model.pt"
    save_model(path, mpnn)
    agg = NormAggregation(norm=50)

    loaded = MPNN.load_from_checkpoint(path, weights_only=True, init_lr=0.5, agg=agg)

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
