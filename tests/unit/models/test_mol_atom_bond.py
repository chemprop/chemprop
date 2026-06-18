import pytest

from chemprop.models import MolAtomBondMPNN
from chemprop.nn import MSE, MABBondMessagePassing, MeanAggregation, RegressionFFN


@pytest.fixture
def mp():
    return MABBondMessagePassing()


@pytest.fixture
def agg():
    return MeanAggregation()


@pytest.fixture
def ffn():
    return RegressionFFN()


@pytest.fixture
def bond_ffn(mp):
    return RegressionFFN(input_dim=mp.output_dims[1] * 2)


@pytest.fixture
def full_model(mp, agg, ffn, bond_ffn):
    return MolAtomBondMPNN(
        message_passing=mp, agg=agg, mol_predictor=ffn, atom_predictor=ffn, bond_predictor=bond_ffn
    )


@pytest.fixture
def partial_model(mp, agg, ffn, bond_ffn):
    return MolAtomBondMPNN(
        message_passing=mp, agg=agg, mol_predictor=ffn, atom_predictor=None, bond_predictor=bond_ffn
    )


def test_output_dimss(full_model, partial_model):
    assert full_model.output_dimss == (1, 1, 1)
    assert partial_model.output_dimss == (1, None, 1)


def test_n_taskss(full_model, partial_model):
    assert full_model.n_taskss == (1, 1, 1)
    assert partial_model.n_taskss == (1, None, 1)


def test_n_targetss(full_model, partial_model):
    assert full_model.n_targetss == (1, 1, 1)
    assert partial_model.n_targetss == (1, None, 1)


def test_criterions_lists(full_model, partial_model):
    assert all(isinstance(c, MSE) for c in full_model.criterions)
    assert isinstance(partial_model.criterions[0], MSE)
    assert partial_model.criterions[1] is None
    assert isinstance(partial_model.criterions[2], MSE)


def test_bond_predictor_dim_validation(mp, agg, ffn):
    """bond_predictor with wrong input_dim should raise ValueError."""
    wrong_bond_ffn = RegressionFFN()  # default input_dim=300, expects 600
    with pytest.raises(ValueError, match="bond_predictor input_dim"):
        MolAtomBondMPNN(
            message_passing=mp, agg=agg, mol_predictor=ffn,
            atom_predictor=ffn, bond_predictor=wrong_bond_ffn
        )
