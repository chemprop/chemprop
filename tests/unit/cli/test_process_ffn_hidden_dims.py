from argparse import Namespace

import pytest

from chemprop.cli.train import _process_ffn_hidden_dims


@pytest.fixture
def base_args():
    return Namespace(
        ffn_hidden_dim=[300],
        ffn_num_layers=1,
        atom_ffn_hidden_dim=[300],
        atom_ffn_num_layers=1,
        bond_ffn_hidden_dim=[300],
        bond_ffn_num_layers=1,
        atom_constrainer_ffn_hidden_dim=[300],
        atom_constrainer_ffn_num_layers=1,
        bond_constrainer_ffn_hidden_dim=[300],
        bond_constrainer_ffn_num_layers=1,
    )


class TestProcessFFNHiddenDims:
    """Tests for _process_ffn_hidden_dims normalization logic."""

    def test_single_int_expands_to_match_n_layers(self, base_args):
        base_args.ffn_hidden_dim = [300]
        base_args.ffn_num_layers = 3
        _process_ffn_hidden_dims(base_args, "ffn_hidden_dim", "ffn_num_layers")
        assert base_args.ffn_hidden_dim == [300, 300, 300]
        assert base_args.ffn_num_layers == 3

    def test_single_int_single_layer_no_change(self, base_args):
        base_args.ffn_hidden_dim = [256]
        base_args.ffn_num_layers = 1
        _process_ffn_hidden_dims(base_args, "ffn_hidden_dim", "ffn_num_layers")
        assert base_args.ffn_hidden_dim == [256]
        assert base_args.ffn_num_layers == 1

    def test_per_layer_explicit_n_layers(self, base_args):
        base_args.ffn_hidden_dim = [600, 300, 150]
        base_args.ffn_num_layers = 3
        _process_ffn_hidden_dims(base_args, "ffn_hidden_dim", "ffn_num_layers")
        assert base_args.ffn_hidden_dim == [600, 300, 150]
        assert base_args.ffn_num_layers == 3

    def test_per_layer_matches_n_layers(self, base_args):
        base_args.ffn_hidden_dim = [512, 64, 512]
        base_args.ffn_num_layers = 3
        _process_ffn_hidden_dims(base_args, "ffn_hidden_dim", "ffn_num_layers")
        assert base_args.ffn_hidden_dim == [512, 64, 512]
        assert base_args.ffn_num_layers == 3

    def test_mismatch_raises_error(self, base_args):
        from configargparse import ArgumentError

        base_args.ffn_hidden_dim = [600, 300]
        base_args.ffn_num_layers = 3
        with pytest.raises(ArgumentError, match="must explicitly pass"):
            _process_ffn_hidden_dims(base_args, "ffn_hidden_dim", "ffn_num_layers")

    def test_all_ffn_variants_processed(self, base_args):
        from chemprop.cli.train import process_train_args

        base_args.ffn_hidden_dim = [600, 300, 150]
        base_args.ffn_num_layers = 3
        processed = process_train_args(base_args)
        assert processed.ffn_hidden_dim == [600, 300, 150]
        assert processed.ffn_num_layers == 3
        assert processed.atom_ffn_hidden_dim == [300]
        assert processed.bond_ffn_hidden_dim == [300]

    def test_per_layer_two_values(self, base_args):
        base_args.ffn_hidden_dim = [256, 128]
        base_args.ffn_num_layers = 2
        _process_ffn_hidden_dims(base_args, "ffn_hidden_dim", "ffn_num_layers")
        assert base_args.ffn_hidden_dim == [256, 128]
        assert base_args.ffn_num_layers == 2

    def test_mismatch_none_raises_error(self, base_args):
        from configargparse import ArgumentError

        base_args.ffn_hidden_dim = [600, 300, 150]
        base_args.ffn_num_layers = 1
        with pytest.raises(ArgumentError, match="must explicitly pass"):
            _process_ffn_hidden_dims(base_args, "ffn_hidden_dim", "ffn_num_layers")
