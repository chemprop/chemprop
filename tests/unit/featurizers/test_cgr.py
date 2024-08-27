import random
from typing import NamedTuple
import uuid

import numpy as np
import pytest

from chemprop.featurizers.molgraph import CGRFeaturizer, RxnMode
from chemprop.utils import make_mol

AVAILABLE_RXN_MODE_NAMES = [
    "REAC_PROD",
    "REAC_PROD_BALANCE",
    "REAC_DIFF",
    "REAC_DIFF_BALANCE",
    "PROD_DIFF",
    "PROD_DIFF_BALANCE",
]


@pytest.fixture
def expected_aliases():
    return AVAILABLE_RXN_MODE_NAMES


@pytest.fixture(params=AVAILABLE_RXN_MODE_NAMES)
def mode_name(request):
    return request.param


@pytest.fixture(params=AVAILABLE_RXN_MODE_NAMES[::2])
def mode_imbalanced(request):
    return request.param


@pytest.fixture(params=AVAILABLE_RXN_MODE_NAMES[1::2])
def mode_balanced(request):
    return request.param


@pytest.fixture
def rxn_mode(mode_name):
    return getattr(RxnMode, mode_name)


@pytest.fixture(params=[str(uuid.uuid4()) for _ in range(3)])
def invalid_alias(request):
    return request.param


rxn_smis = [
    # reactant and product with the same number of atoms
    "[CH3:1][H:2]>>[CH3:1].[H:2]",  # reactant and product are balanced and mapped
    "[CH3:2][H:1]>>[H:1].[CH3:2]",  # reactant and product are balanced, mapped but with different atom index order
    "[CH3:1][H]>>[CH3:1].[H:2]",  # reactant and product are balanced and but reactant has less atom-mapped atoms
    "[CH3:1][H:2]>>[H].[CH3:1]",  # reactant and product are balanced and but product has less atom-mapped atoms
    # reactant and product has different numbers of atoms
    "[CH4:1]>>[CH2:1].[H:2][H:3]",  # product has more atoms and more atom-mapped atoms
    "[H:1].[CH2:2][H:3]>>[CH3:2][H:3]",  # reactant with more atoms and atom-mapped atoms
    "[CH4:1]>>[CH3:1].[H:2]",  # product with more atoms and atom-mapped atoms with 0 edge
]

# Expected output for CGRFeaturizer.map_reac_to_prod
reac_prod_maps = {
    "[CH3:1][H:2]>>[CH3:1].[H:2]": ({0: 0, 1: 1}, [], []),
    "[CH3:2][H:1]>>[H:1].[CH3:2]": ({0: 1, 1: 0}, [], []),
    "[CH3:1][H]>>[CH3:1].[H:2]": ({0: 0}, [1], [1]),
    "[CH3:1][H:2]>>[H].[CH3:1]": ({0: 1}, [0], [1]),
    "[CH4:1]>>[CH2:1].[H:2][H:3]": ({0: 0}, [1, 2], []),
    "[H:1].[CH2:2][H:3]>>[CH3:2][H:3]": ({1: 0, 2: 1}, [], [0]),
    "[CH4:1]>>[CH3:1].[H:2]": ({0: 0}, [1], []),
}


@pytest.fixture(params=rxn_smis)
def rxn_smi(request):
    return request.param


class BondExpectation(NamedTuple):
    """
    whether elements in the returns for _get_bonds are Nones under
    imbalanced and balanced modes for provided bond
    """

    bond: tuple
    bond_reac_none: bool
    bond_prod_none: bool


bond_expect_imbalanced = {
    "[CH3:1][H:2]>>[CH3:1].[H:2]": [
        BondExpectation((0, 1), bond_reac_none=False, bond_prod_none=True)
    ],
    "[CH3:2][H:1]>>[H:1].[CH3:2]": [
        BondExpectation((0, 1), bond_reac_none=False, bond_prod_none=True)
    ],
    "[CH3:1][H]>>[CH3:1].[H:2]": [
        BondExpectation((0, 1), bond_reac_none=False, bond_prod_none=True),
        BondExpectation((0, 2), bond_reac_none=True, bond_prod_none=True),
        BondExpectation((1, 2), bond_reac_none=True, bond_prod_none=True),
    ],
    "[CH3:1][H:2]>>[H].[CH3:1]": [
        BondExpectation((0, 1), bond_reac_none=False, bond_prod_none=True),
        BondExpectation((0, 2), bond_reac_none=True, bond_prod_none=True),
        BondExpectation((1, 2), bond_reac_none=True, bond_prod_none=True),
    ],
    "[CH4:1]>>[CH2:1].[H:2][H:3]": [
        BondExpectation((0, 1), bond_reac_none=True, bond_prod_none=True),
        BondExpectation((0, 2), bond_reac_none=True, bond_prod_none=True),
        BondExpectation((1, 2), bond_reac_none=True, bond_prod_none=False),
    ],
    "[H:1].[CH2:2][H:3]>>[CH3:2][H:3]": [
        BondExpectation((0, 1), bond_reac_none=True, bond_prod_none=True),
        BondExpectation((0, 2), bond_reac_none=True, bond_prod_none=True),
        BondExpectation((1, 2), bond_reac_none=False, bond_prod_none=False),
    ],
    "[CH4:1]>>[CH3:1].[H:2]": [
        BondExpectation((0, 0), bond_reac_none=True, bond_prod_none=True)
    ],  # this last entry doesn't test for anything meaningful, only to enable other tests for graph with zero edges
}
bond_expect_balanced = bond_expect_imbalanced.copy()
bond_expect_balanced.update(
    {
        "[CH4:1]>>[CH2:1].[H:2][H:3]": [
            BondExpectation((0, 1), bond_reac_none=True, bond_prod_none=True),
            BondExpectation((0, 2), bond_reac_none=True, bond_prod_none=True),
            BondExpectation((1, 2), bond_reac_none=False, bond_prod_none=False),
        ]  # this is the only difference compared to the imbalanced case
    }
)


# A fake `bond` is used in test_calc_edge_features. This is a workaround,
# as RDKit cannot construct a bond directly in Python
bond = make_mol("[CH3:1][H:2]", keep_h=True, add_h=False).GetBondWithIdx(0)


def get_reac_prod(rxn_smi: str) -> list:
    return [make_mol(smi, keep_h=True, add_h=False) for smi in rxn_smi.split(">>")]


def randomize_case(s: str) -> str:
    choices = (str.upper, str.lower)

    return "".join(random.choice(choices)(x) for x in s)


@pytest.mark.parametrize("s", [str(uuid.uuid4()) for _ in range(3)])
def test_randomize_case(s):
    """test our helper function to ensure that it's not mangling our strings"""
    assert randomize_case(s).upper() == s.upper()


def test_len(expected_aliases):
    """
    Test that the RxnMode class has the correct length.
    """
    assert len(RxnMode) == len(expected_aliases)


def test_keys(expected_aliases):
    """
    Test that the keys function returns the correct set of modes.
    """
    assert set(RxnMode.keys()) == set(alias.upper() for alias in expected_aliases)


@pytest.mark.parametrize(
    "alias,rxn_mode",
    [
        ("REAC_PROD", RxnMode.REAC_PROD),
        ("REAC_PROD_BALANCE", RxnMode.REAC_PROD_BALANCE),
        ("REAC_DIFF", RxnMode.REAC_DIFF),
        ("REAC_DIFF_BALANCE", RxnMode.REAC_DIFF_BALANCE),
        ("PROD_DIFF", RxnMode.PROD_DIFF),
        ("PROD_DIFF_BALANCE", RxnMode.PROD_DIFF_BALANCE),
    ],
)
class TestRxnModeGet:
    def test_name_and_value(self, alias, rxn_mode):
        assert alias.upper() == rxn_mode.name
        assert alias.lower() == rxn_mode.value

    def test_getitem(self, alias, rxn_mode):
        """
        Test that the RxnMode class can be indexed with uppercase mode.
        """
        assert RxnMode[alias.upper()] == rxn_mode

    def test_get(self, alias, rxn_mode):
        """
        Test that the get function returns the correct RxnMode.
        """
        assert RxnMode.get(alias.upper()) == rxn_mode

    def test_get_random_case(self, alias, rxn_mode):
        """
        Test that the get function returns the correct RxnMode when given an alias with random case.
        """
        assert RxnMode.get(randomize_case(alias)) == rxn_mode

    def test_get_enum_identity(self, alias, rxn_mode):
        """
        Test that the get function returns the correct RxnMode when given a RxnMode.
        """
        assert RxnMode.get(rxn_mode) == rxn_mode


def test_getitem_invalid_mode(invalid_alias):
    """
    Test that the RxnMode class raises a ValueError when indexed with an invalid mode.
    """
    with pytest.raises(KeyError):
        RxnMode[invalid_alias]


def test_get_invalid_mode(invalid_alias):
    """
    Test that the get function raises a ValueError when given an invalid mode.
    """
    with pytest.raises(KeyError):
        RxnMode.get(invalid_alias)


class TestCondensedGraphOfReactionFeaturizer:
    def test_init_without_mode_(self):
        """
        Test that the CondensedGraphOfReactionFeaturizer can be initialized without a mode.
        """
        featurizer = CGRFeaturizer()
        assert featurizer.mode == RxnMode.REAC_DIFF

    def test_init_with_mode_str(self, mode_name, rxn_mode):
        """
        Test that the CondensedGraphOfReactionFeaturizer can be initialized with a string of the mode.
        """
        featurizer = CGRFeaturizer(mode_=mode_name)
        assert featurizer.mode == rxn_mode

    def test_init_with_mode_enum(self, rxn_mode):
        """
        Test that the CondensedGraphOfReactionFeaturizer can be initialized with a RxnMode.
        """
        featurizer = CGRFeaturizer(mode_=rxn_mode)
        assert featurizer.mode == rxn_mode

    def test_map_reac_to_prod(self, rxn_smi):
        """
        Test that the map_reac_to_prod method returns the correct mapping.
        """
        reac, prod = get_reac_prod(rxn_smi)
        assert CGRFeaturizer.map_reac_to_prod(reac, prod) == reac_prod_maps[rxn_smi]

    def test_calc_node_feature_matrix_shape(self, rxn_smi, mode_name):
        """
        Test that the calc_node_feature_matrix method returns the correct node feature matrix.
        """
        featurizer = CGRFeaturizer(mode_=mode_name)

        reac, prod = get_reac_prod(rxn_smi)
        ri2pj, pids, rids = featurizer.map_reac_to_prod(reac, prod)

        num_nodes, atom_fdim = featurizer._calc_node_feature_matrix(
            reac, prod, ri2pj, pids, rids
        ).shape
        assert num_nodes == len(ri2pj) + len(pids) + len(rids)
        assert atom_fdim == featurizer.atom_fdim

    def test_calc_node_feature_matrix_atomic_number_features(self, rxn_smi, rxn_mode):
        """
        Test that the calc_node_feature_matrix method returns the correct feature matrix for the atomic number features.
        """
        featurizer = CGRFeaturizer(mode_=rxn_mode)
        reac, prod = get_reac_prod(rxn_smi)
        ri2pj, pids, rids = featurizer.map_reac_to_prod(reac, prod)
        atom_featurizer = featurizer.atom_featurizer

        atomic_num_features_expected = np.array(
            [atom_featurizer.num_only(a) for a in reac.GetAtoms()]
            + [atom_featurizer.num_only(prod.GetAtomWithIdx(pid)) for pid in pids]
        )[
            :, : len(atom_featurizer.atomic_nums) + 1
        ]  # only create and keep the atomic number features

        atomic_num_features = featurizer._calc_node_feature_matrix(reac, prod, ri2pj, pids, rids)[
            :, : len(atom_featurizer.atomic_nums) + 1
        ]

        np.testing.assert_equal(atomic_num_features, atomic_num_features_expected)

    def test_get_bonds_imbalanced(self, rxn_smi, mode_imbalanced):
        """
        Test that the get_bonds method returns the correct bonds when modes are imbalanced.
        """
        featurizer = CGRFeaturizer(mode_=mode_imbalanced)
        reac, prod = get_reac_prod(rxn_smi)
        ri2pj, pids, _ = featurizer.map_reac_to_prod(reac, prod)

        for bond_expect in bond_expect_imbalanced[rxn_smi]:
            bond_reac, bond_prod = featurizer._get_bonds(
                reac, prod, ri2pj, pids, reac.GetNumAtoms(), *bond_expect.bond
            )
            assert (bond_reac is None) == bond_expect.bond_reac_none
            assert (bond_prod is None) == bond_expect.bond_prod_none

    def test_get_bonds_balanced(self, rxn_smi, mode_balanced):
        """
        Test that the get_bonds method returns the correct bonds when modes are balanced.
        """
        featurizer = CGRFeaturizer(mode_=mode_balanced)
        reac, prod = get_reac_prod(rxn_smi)
        ri2pj, pids, _ = featurizer.map_reac_to_prod(reac, prod)

        for bond_expect in bond_expect_balanced[rxn_smi]:
            bond_reac, bond_prod = featurizer._get_bonds(
                reac, prod, ri2pj, pids, reac.GetNumAtoms(), *bond_expect.bond
            )
            assert (bond_reac is None) == bond_expect.bond_reac_none
            assert (bond_prod is None) == bond_expect.bond_prod_none

    @pytest.mark.parametrize(
        "reac_prod_bonds", [(bond, bond), (bond, None), (None, bond), (None, None)]
    )
    def test_calc_edge_feature_shape(self, reac_prod_bonds, rxn_mode):
        """
        Test that the calc_edge_feature method returns the correct edge feature.
        """
        featurizer = CGRFeaturizer(mode_=rxn_mode)
        reac_bond, prod_bond = reac_prod_bonds

        assert featurizer._calc_edge_feature(reac_bond, prod_bond).shape == (
            len(featurizer.bond_featurizer) * 2,
        )

    def test_featurize_balanced(self, rxn_smi, mode_balanced):
        """
        Test CGR featurizer returns the correct features with balanced modes.
        """
        featurizer = CGRFeaturizer(mode_=mode_balanced)
        reac, prod = get_reac_prod(rxn_smi)
        ri2pj, pids, rids = featurizer.map_reac_to_prod(reac, prod)

        molgraph = featurizer((reac, prod))

        n_atoms = len(ri2pj) + len(pids) + len(rids)
        atom_fdim = featurizer.atom_fdim

        assert molgraph.V.shape == (n_atoms, atom_fdim)

        bonds = [
            b.bond
            for b in bond_expect_balanced[rxn_smi]
            if not (b.bond_reac_none and b.bond_prod_none)
        ]
        bond_fdim = featurizer.bond_fdim

        assert molgraph.E.shape == (len(bonds) * 2, bond_fdim)

        expect_edge_index = [[], []]
        expect_rev_edge_index = []

        for i, bond in enumerate(bonds):
            bond = list(bond)
            expect_edge_index[0].extend(bond)
            expect_edge_index[1].extend(bond[::-1])
            expect_rev_edge_index.extend([i * 2 + 1, i * 2])

        assert np.array_equal(molgraph.edge_index, expect_edge_index)
        assert np.array_equal(molgraph.rev_edge_index, expect_rev_edge_index)

    def test_featurize_imbalanced(self, rxn_smi, mode_imbalanced):
        """
        Test CGR featurizer returns the correct features with balanced modes.
        """
        featurizer = CGRFeaturizer(mode_=mode_imbalanced)
        reac, prod = get_reac_prod(rxn_smi)
        ri2pj, pids, rids = featurizer.map_reac_to_prod(reac, prod)

        molgraph = featurizer((reac, prod))

        n_atoms = len(ri2pj) + len(pids) + len(rids)
        atom_fdim = featurizer.atom_fdim

        assert molgraph.V.shape == (n_atoms, atom_fdim)

        bonds = [
            b.bond
            for b in bond_expect_imbalanced[rxn_smi]
            if not (b.bond_reac_none and b.bond_prod_none)
        ]
        bond_fdim = featurizer.bond_fdim

        assert molgraph.E.shape == (len(bonds) * 2, bond_fdim)

        expect_edge_index = [[], []]
        expect_rev_edge_index = []

        for i, bond in enumerate(bonds):
            bond = list(bond)
            expect_edge_index[0].extend(bond)
            expect_edge_index[1].extend(bond[::-1])
            expect_rev_edge_index.extend([i * 2 + 1, i * 2])

        assert np.array_equal(molgraph.edge_index, expect_edge_index)
        assert np.array_equal(molgraph.rev_edge_index, expect_rev_edge_index)
