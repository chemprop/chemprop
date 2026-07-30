import numpy as np
import pandas as pd
import pytest

from chemprop.data.collate import BatchMolGraph
from chemprop.featurizers.atom import get_multi_hot_atom_featurizer
from chemprop.featurizers.bond import MultiHotBondFeaturizer, RIGRBondFeaturizer
from chemprop.featurizers.molgraph import (
    CondensedGraphOfReactionFeaturizer,
    CuikmolmakerCGRFeaturizer,
    RxnMode,
)
from chemprop.utils.utils import make_mol


@pytest.fixture(params=["V2", "V1", "ORGANIC", "RIGR"])
def atom_featurizer_mode(request):
    return request.param


@pytest.fixture(params=list(RxnMode.keys()))
def reaction_mode(request):
    return request.param


@pytest.fixture
def rct_pdt_smis(data_dir):
    df = pd.read_csv(data_dir / "regression/rxn/rxn.csv")
    pairs = []
    for rxn in df["smiles"].tolist()[:10]:
        rct_smi, agt_smi, pdt_smi = rxn.split(">")
        rct_smi = f"{rct_smi}.{agt_smi}" if agt_smi else rct_smi
        pairs.append((rct_smi, pdt_smi))
    return pairs


@pytest.fixture
def python_cgr_featurizer(atom_featurizer_mode, reaction_mode):
    atom_featurizer = get_multi_hot_atom_featurizer(atom_featurizer_mode)
    if atom_featurizer_mode == "RIGR":
        bond_featurizer = RIGRBondFeaturizer()
    else:
        bond_featurizer = MultiHotBondFeaturizer()
    return CondensedGraphOfReactionFeaturizer(
        atom_featurizer=atom_featurizer, bond_featurizer=bond_featurizer, mode_=reaction_mode
    )


@pytest.fixture
def cuik_reaction_featurizer(atom_featurizer_mode, reaction_mode):
    return CuikmolmakerCGRFeaturizer(
        atom_featurizer_mode=atom_featurizer_mode, reaction_mode=reaction_mode, keep_h=True
    )


@pytest.fixture
def bmg_python_cgr(rct_pdt_smis, python_cgr_featurizer):
    mgs = [
        python_cgr_featurizer((make_mol(rct_smi, keep_h=True), make_mol(pdt_smi, keep_h=True)))
        for rct_smi, pdt_smi in rct_pdt_smis
    ]
    return BatchMolGraph(mgs)


@pytest.fixture
def bmg_cuik_reaction(rct_pdt_smis, cuik_reaction_featurizer):
    rct_smis = [rct_smi for rct_smi, _ in rct_pdt_smis]
    pdt_smis = [pdt_smi for _, pdt_smi in rct_pdt_smis]
    return cuik_reaction_featurizer(rct_smis, pdt_smis)


def test_fdim_matches(python_cgr_featurizer, cuik_reaction_featurizer):
    assert cuik_reaction_featurizer.atom_fdim == python_cgr_featurizer.atom_fdim
    assert cuik_reaction_featurizer.bond_fdim == python_cgr_featurizer.bond_fdim


def test_same_featurization(bmg_python_cgr, bmg_cuik_reaction):
    np.testing.assert_allclose(bmg_python_cgr.V, bmg_cuik_reaction.V.numpy())
    np.testing.assert_allclose(bmg_python_cgr.E, bmg_cuik_reaction.E.numpy())
    np.testing.assert_allclose(bmg_python_cgr.edge_index, bmg_cuik_reaction.edge_index.numpy())
    np.testing.assert_allclose(
        bmg_python_cgr.rev_edge_index, bmg_cuik_reaction.rev_edge_index.numpy()
    )
    np.testing.assert_allclose(bmg_python_cgr.batch, bmg_cuik_reaction.batch.numpy())
