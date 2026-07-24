"""Parity tests between the pure-python and cuik-molmaker MolAtomBond featurization paths.

Atom and bond targets are matched positionally to rows of ``V``/``E``, so an ordering mismatch
between the two featurizers would train silently on misaligned data. These tests pin that ordering.
"""

import numpy as np
import pytest
import torch

from chemprop.data.collate import collate_cuik_mol_atom_bond_batch, collate_mol_atom_bond_batch
from chemprop.data.datapoints import LazyMolAtomBondDatapoint, MolAtomBondDatapoint
from chemprop.data.datasets import CuikmolmakerMolAtomBondDataset, MolAtomBondDataset
from chemprop.featurizers.molgraph import (
    CuikmolmakerMolGraphFeaturizer,
    SimpleMoleculeMolGraphFeaturizer,
)
from chemprop.utils import make_mol

# "C" (methane) has zero bonds and "CC" (ethane) has exactly one, the cases most likely to break
# per-bond batching (`bond_batch`, weights repeated per bond). `test_edge_case_smis_coverage`
# guards this so the coverage can't be dropped by a later edit.
EDGE_CASE_SMIS = ["C", "CC", "CCO", "c1ccccc1", "CC(=O)O"]

EXTRA_ATOM_FDIM = 3
EXTRA_BOND_FDIM = 2


def test_edge_case_smis_coverage():
    n_bondss = {make_mol(smi, False, False, False, False).GetNumBonds() for smi in EDGE_CASE_SMIS}
    assert {0, 1} <= n_bondss, "parity fixtures must keep a zero-bond and a single-bond molecule"


@pytest.fixture(params=[EDGE_CASE_SMIS, None], ids=["edge_case_smis", "sampled_smis"])
def smis_list(smis, request):
    return request.param if request.param is not None else smis.sample(6).to_list()


@pytest.fixture
def mols(smis_list):
    return [make_mol(smi, False, False, False, False) for smi in smis_list]


@pytest.fixture
def targets(mols):
    rng = np.random.default_rng(0)
    return (
        rng.random((len(mols), 2)),
        [rng.random((mol.GetNumAtoms(), 2)) for mol in mols],
        [rng.random((mol.GetNumBonds(), 2)) for mol in mols],
    )


@pytest.fixture
def weights(mols):
    return np.random.default_rng(1).random(len(mols))


@pytest.fixture(params=[False, True], ids=["no_extras", "extras"])
def extras(mols, request):
    """The extra features/descriptors that ride alongside the graph. ``E_d`` is duplicated per
    bond direction, mirroring what ``build_MAB_data_from_files`` does at parse time."""
    if not request.param:
        return [dict() for _ in mols]

    rng = np.random.default_rng(2)
    return [
        dict(
            x_d=rng.random(2),
            V_f=rng.random((mol.GetNumAtoms(), EXTRA_ATOM_FDIM)),
            E_f=rng.random((mol.GetNumBonds(), EXTRA_BOND_FDIM)),
            V_d=rng.random((mol.GetNumAtoms(), 4)),
            E_d=np.repeat(rng.random((mol.GetNumBonds(), 5)), repeats=2, axis=0),
        )
        for mol in mols
    ]


@pytest.fixture
def datapoint_kwargs(targets, weights, extras):
    mol_ys, atom_ys, bond_ys = targets
    return [
        dict(
            y=mol_ys[i], atom_y=atom_ys[i], bond_y=bond_ys[i], weight=float(weights[i]), **extras[i]
        )
        for i in range(len(weights))
    ]


@pytest.fixture
def pp_dataset(mols, datapoint_kwargs, extras):
    data = [MolAtomBondDatapoint(mol=mol, **datapoint_kwargs[i]) for i, mol in enumerate(mols)]
    featurizer = SimpleMoleculeMolGraphFeaturizer(
        extra_atom_fdim=EXTRA_ATOM_FDIM if "V_f" in extras[0] else 0,
        extra_bond_fdim=EXTRA_BOND_FDIM if "E_f" in extras[0] else 0,
    )
    return MolAtomBondDataset(data, featurizer)


@pytest.fixture
def cuik_dataset(smis_list, datapoint_kwargs, extras):
    data = [
        LazyMolAtomBondDatapoint(smiles=smi, **datapoint_kwargs[i])
        for i, smi in enumerate(smis_list)
    ]
    featurizer = CuikmolmakerMolGraphFeaturizer(
        atom_featurizer_mode="V2",
        extra_atom_fdim=EXTRA_ATOM_FDIM if "V_f" in extras[0] else 0,
        extra_bond_fdim=EXTRA_BOND_FDIM if "E_f" in extras[0] else 0,
    )
    return CuikmolmakerMolAtomBondDataset(data, featurizer)


def _assert_optional_close(a, b):
    if a is None or b is None:
        assert a is None and b is None
    else:
        torch.testing.assert_close(a, b)


def _assert_batches_match(pp_batch, cuik_batch):
    torch.testing.assert_close(pp_batch.bmg.V, cuik_batch.bmg.V)
    torch.testing.assert_close(pp_batch.bmg.E, cuik_batch.bmg.E)
    assert torch.equal(pp_batch.bmg.edge_index, cuik_batch.bmg.edge_index)
    assert torch.equal(pp_batch.bmg.rev_edge_index, cuik_batch.bmg.rev_edge_index)
    assert torch.equal(pp_batch.bmg.batch, cuik_batch.bmg.batch)
    assert torch.equal(pp_batch.bmg.bond_batch, cuik_batch.bmg.bond_batch)

    _assert_optional_close(pp_batch.V_d, cuik_batch.V_d)
    _assert_optional_close(pp_batch.E_d, cuik_batch.E_d)
    _assert_optional_close(pp_batch.X_d, cuik_batch.X_d)

    for attr in ("Ys", "w", "lt_masks", "gt_masks"):
        for pp_x, cuik_x in zip(getattr(pp_batch, attr), getattr(cuik_batch, attr)):
            _assert_optional_close(pp_x, cuik_x)


def _collate_both(pp_dataset, cuik_dataset):
    indexes = list(range(len(pp_dataset)))
    return (
        collate_mol_atom_bond_batch([pp_dataset[i] for i in indexes]),
        collate_cuik_mol_atom_bond_batch(cuik_dataset.__getitems__(indexes)),
    )


def test_parity_full_batch(pp_dataset, cuik_dataset):
    _assert_batches_match(*_collate_both(pp_dataset, cuik_dataset))


def test_single_item_getitem_parity(pp_dataset, cuik_dataset):
    for idx in range(len(pp_dataset)):
        pp_item = pp_dataset[idx]
        cuik_item = cuik_dataset[idx]

        np.testing.assert_allclose(pp_item.mg.V, cuik_item.mg.V)
        np.testing.assert_allclose(pp_item.mg.E, cuik_item.mg.E)
        np.testing.assert_array_equal(pp_item.mg.edge_index, cuik_item.mg.edge_index)
        np.testing.assert_array_equal(pp_item.mg.rev_edge_index, cuik_item.mg.rev_edge_index)


def test_add_h_parity():
    """cuik-molmaker appends hydrogens internally when ``add_h=True``; their order must match
    RDKit's own ``AddHs`` order for atom targets to stay aligned with features."""
    smis_list = ["CCO", "c1ccccc1", "C"]
    mols = [make_mol(smi, False, True, False, False) for smi in smis_list]

    rng = np.random.default_rng(42)
    kwargs = [
        dict(
            y=rng.random(2),
            atom_y=rng.random((mol.GetNumAtoms(), 2)),
            bond_y=rng.random((mol.GetNumBonds(), 2)),
            weight=float(rng.random()),
        )
        for mol in mols
    ]

    pp_dataset = MolAtomBondDataset(
        [MolAtomBondDatapoint(mol=mol, **kwargs[i]) for i, mol in enumerate(mols)],
        SimpleMoleculeMolGraphFeaturizer(),
    )
    cuik_dataset = CuikmolmakerMolAtomBondDataset(
        [
            LazyMolAtomBondDatapoint(smiles=smi, _add_h=True, **kwargs[i])
            for i, smi in enumerate(smis_list)
        ],
        CuikmolmakerMolGraphFeaturizer(atom_featurizer_mode="V2", add_h=True),
    )

    _assert_batches_match(*_collate_both(pp_dataset, cuik_dataset))


def test_cache_is_rejected(cuik_dataset):
    with pytest.raises(NotImplementedError, match="without caching"):
        cuik_dataset.cache = True


def test_smiles_returns_input_verbatim():
    """`MoleculeDataset.smiles` round-trips through `Chem.MolToSmiles`, which canonicalizes and so
    would reorder atoms relative to the atom targets. The cuik dataset must hand the *input* SMILES
    to the featurizer instead. These inputs are deliberately non-canonical ('OCC' -> 'CCO'), so
    this fails if the override is dropped."""
    smis_list = ["OCC", "C(C)O"]
    dataset = CuikmolmakerMolAtomBondDataset(
        [LazyMolAtomBondDatapoint(smiles=smi) for smi in smis_list],
        CuikmolmakerMolGraphFeaturizer(atom_featurizer_mode="V2"),
    )

    assert dataset.smiles == smis_list
