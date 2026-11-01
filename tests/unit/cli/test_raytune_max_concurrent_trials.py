"""Regression for chemprop/chemprop#1402: wire --raytune-max-concurrent-trials into TuneConfig."""

from argparse import Namespace
from unittest.mock import MagicMock, patch

import pytest

from chemprop.cli.hpopt import NO_RAY, process_hpopt_args, tune_model


pytestmark = pytest.mark.skipif(NO_RAY, reason="Ray not installed")


def _minimal_hpopt_args(**overrides) -> Namespace:
    args = Namespace(
        search_parameter_keywords=["depth"],
        epochs=2,
        tracking_metric="val_loss",
        raytune_trial_scheduler="FIFO",
        raytune_grace_period=10,
        raytune_reduction_factor=2,
        raytune_num_cpus=None,
        raytune_num_gpus=None,
        raytune_max_concurrent_trials=1,
        raytune_num_workers=1,
        raytune_use_gpu=False,
        raytune_num_checkpoints_to_keep=1,
        raytune_num_samples=2,
        raytune_search_algorithm="random",
        hpopt_save_dir=MagicMock(),
        hyperopt_n_initial_points=None,
        hyperopt_random_state_seed=None,
        data_path=[MagicMock(stem="data")],
        mol_target_columns=None,
        atom_target_columns=None,
        bond_target_columns=None,
        from_foundation=None,
        constraints_path=None,
    )
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


def test_process_hpopt_args_rejects_non_positive_max_concurrent_trials(tmp_path):
    args = _minimal_hpopt_args(
        raytune_max_concurrent_trials=0,
        hpopt_save_dir=tmp_path / "hpopt",
        data_path=[MagicMock(stem="data")],
    )
    with pytest.raises(ValueError, match="--raytune-max-concurrent-trials must be >= 1"):
        process_hpopt_args(args)


def test_tune_model_passes_max_concurrent_trials_to_tune_config(tmp_path):
    """Ensure max_concurrent_trials=1 reaches TuneConfig (previously dropped; #1402)."""
    args = _minimal_hpopt_args(hpopt_save_dir=tmp_path)

    captured = {}

    class FakeTuneConfig:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    fake_tuner = MagicMock()
    fake_tuner.fit.return_value = "results"

    with (
        patch("chemprop.cli.hpopt.tune.TuneConfig", FakeTuneConfig),
        patch("chemprop.cli.hpopt.tune.Tuner", return_value=fake_tuner) as mock_tuner,
        patch("chemprop.cli.hpopt.build_search_space", return_value={"depth": 3}),
        patch("chemprop.cli.hpopt.TorchTrainer"),
        patch("chemprop.cli.hpopt.ScalingConfig"),
        patch("chemprop.cli.hpopt.CheckpointConfig"),
        patch("chemprop.cli.hpopt.RunConfig"),
        patch("chemprop.cli.hpopt.FIFOScheduler"),
    ):
        result = tune_model(
            args,
            train_dset=MagicMock(),
            val_dset=MagicMock(),
            logger=MagicMock(),
            monitor_mode="min",
            output_transform=None,
            input_transforms=None,
        )

    assert result == "results"
    assert captured["max_concurrent_trials"] == 1
    assert captured["num_samples"] == 2
    mock_tuner.assert_called_once()
    assert isinstance(mock_tuner.call_args.kwargs["tune_config"], FakeTuneConfig)