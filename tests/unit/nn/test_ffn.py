"""Tests for :class:`chemprop.nn.ffn.MLP` layer sizing."""

import pytest
import torch
from torch import nn

from chemprop.nn.ffn import MLP
from chemprop.nn.predictors import RegressionFFN


def hidden_widths(mlp: MLP) -> list[int]:
    """The out_features of every Linear except the output layer."""
    linears = [m for block in mlp for m in block if isinstance(m, nn.Linear)]
    return [lin.out_features for lin in linears[:-1]]


@pytest.mark.parametrize("n_layers", [0, 1, 2, 5])
def test_int_hidden_dim_repeats(n_layers):
    mlp = MLP.build(input_dim=8, output_dim=2, hidden_dim=64, n_layers=n_layers)

    assert hidden_widths(mlp) == [64] * n_layers
    assert mlp.input_dim == 8
    assert mlp.output_dim == 2


def test_sequence_hidden_dim_sets_each_layer():
    mlp = MLP.build(input_dim=8, output_dim=2, hidden_dim=[32, 16, 4])

    assert hidden_widths(mlp) == [32, 16, 4]
    assert mlp.input_dim == 8
    assert mlp.output_dim == 2


def test_sequence_hidden_dim_ignores_n_layers():
    """The sequence length is the depth, so n_layers cannot contradict it."""
    mlp = MLP.build(input_dim=8, output_dim=2, hidden_dim=[32, 16], n_layers=7)

    assert hidden_widths(mlp) == [32, 16]


@pytest.mark.parametrize("hidden_dim", [(32, 16), [32, 16]])
def test_any_sequence_type_is_accepted(hidden_dim):
    assert hidden_widths(MLP.build(input_dim=8, output_dim=2, hidden_dim=hidden_dim)) == [32, 16]


def test_empty_sequence_gives_a_linear_model():
    mlp = MLP.build(input_dim=8, output_dim=2, hidden_dim=[])

    assert hidden_widths(mlp) == []
    assert mlp.input_dim == 8
    assert mlp.output_dim == 2


def test_int_and_sequence_agree_when_widths_match():
    """An int is exactly the constant-width case of a sequence."""
    from_int = MLP.build(input_dim=8, output_dim=2, hidden_dim=64, n_layers=3)
    from_seq = MLP.build(input_dim=8, output_dim=2, hidden_dim=[64, 64, 64])

    assert hidden_widths(from_int) == hidden_widths(from_seq)


def test_forward_pass_with_varying_widths():
    mlp = MLP.build(input_dim=8, output_dim=2, hidden_dim=[32, 16, 4])

    assert mlp(torch.randn(5, 8)).shape == (5, 2)


def test_predictor_accepts_a_sequence_hidden_dim():
    ffn = RegressionFFN(n_tasks=3, input_dim=8, hidden_dim=[32, 16])

    assert hidden_widths(ffn.ffn) == [32, 16]
    assert ffn.output_dim == 3 * RegressionFFN.n_targets
    assert ffn(torch.randn(5, 8)).shape[0] == 5


def test_predictor_sequence_hidden_dim_survives_hparams_roundtrip():
    """hidden_dim is saved to hparams, so a checkpoint must reload the same shape."""
    ffn = RegressionFFN(n_tasks=1, input_dim=8, hidden_dim=[32, 16])
    rebuilt = RegressionFFN(
        **{
            k: v
            for k, v in ffn.hparams.items()
            if k not in ("cls", "criterion", "output_transform")
        }
    )

    assert hidden_widths(rebuilt.ffn) == [32, 16]
