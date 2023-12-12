import random
import uuid

import numpy as np
import pytest

from chemprop.v2.featurizers.molgraph import RxnMode, CondensedGraphOfReactionFeaturizer
from chemprop.v2.utils import make_mol


AVAILABLE_RXN_MODE_NAMES = [
    "REAC_PROD",
    "REAC_PROD_BALANCE",
    "REAC_DIFF",
    "REAC_DIFF_BALANCE",
    "PROD_DIFF",
    "PROD_DIFF_BALANCE",
]


# @pytest.fixture(params=list(RxnMode))
# def rxn_mode(request):
#     return request.param


@pytest.fixture
def expected_aliases():
    return AVAILABLE_RXN_MODE_NAMES


@pytest.fixture(params=AVAILABLE_RXN_MODE_NAMES)
def mode_name(request):
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
    "[CH3:1][H]>> [CH3:1].[H:2]",  # reactant and product are balanced and but reactant has less atom-mapped atoms
    "[CH3:1][H:2]>>[H].[CH3:1]",  # reactant and product are balanced and but product has less atom-mapped atoms
    # reactant and product has different numbers of atoms
    "[CH4:1]>>[CH2:1].[H:2][H:3]",  # product has more atoms and more atom-mapped atoms
    "[H:1].[CH2:2][H:3]>>[CH3:2][H:3]",  # reactant with more atoms and atom-mapped atoms.
]

# Expected output for map_reac_to_prod
# It follows the order of rxn_smis
# Note, the sum of the lengths of the three elements equal to
# the number of unique atoms in the reactant and product
reac_prod_maps = [
    ({0: 0, 1: 1}, [], []),
    ({0: 1, 1: 0}, [], []),
    ({0: 0}, [1], [1]),
    ({0: 1}, [0], [1]),
    ({0: 0}, [1, 2], []),
    ({1: 0, 2: 1}, [], [0]),
]


# whether elements in the returns for _get_bonds are Nones under imbalanced and balanced modes
# It follows the order of rxn_smis
# Note, it also includes all the bonding information in the reaction
elements_from_get_bond_is_None_imbalanced = [
    {(0, 1): (False, True)},
    {(0, 1): (False, True)},
    {(0, 1): (False, True), (0, 2): (True, True), (1, 2): (True, True)},
    {(0, 1): (False, True), (0, 2): (True, True), (1, 2): (True, True)},
    {(0, 1): (True, True), (0, 2): (True, True), (1, 2): (True, False)},
    {(0, 1): (True, True), (0, 2): (True, True), (1, 2): (False, False)},
]


elements_from_get_bond_is_None_balanced = [
    {(0, 1): (False, True)},
    {(0, 1): (False, True)},
    {(0, 1): (False, True), (0, 2): (True, True), (1, 2): (True, True)},
    {(0, 1): (False, True), (0, 2): (True, True), (1, 2): (True, True)},
    {
        (0, 1): (True, True),
        (0, 2): (True, True),
        (1, 2): (False, False),
    },  # this is different from the imbalanced case
    {(0, 1): (True, True), (0, 2): (True, True), (1, 2): (False, False)},
]


@pytest.fixture
def reac_prod_mols(request):
    return tuple(make_mol(smi, keep_h=True, add_h=False) for smi in request.param.split(">>"))


@pytest.fixture
def cgr_featurizer(request):
    return CondensedGraphOfReactionFeaturizer(mode_=request.param)


@pytest.fixture(params=[(False, False), (True, False), (False, True), (True, True)])
def bond_reac_prod(request):
    bond = make_mol("[CH3:1][H:2]", keep_h=True, add_h=False).GetBondBetweenAtoms(0, 1)
    return (bond if request.param[0] else None, bond if request.param[1] else None)


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
        cgr_featurizer = CondensedGraphOfReactionFeaturizer()
        assert cgr_featurizer.mode == RxnMode.REAC_DIFF

    def test_init_with_mode_str(self, mode_name):
        """
        Test that the CondensedGraphOfReactionFeaturizer can be initialized with a string of the mode.
        """
        cgr_featurizer = CondensedGraphOfReactionFeaturizer(mode_=mode_name)
        assert cgr_featurizer.mode == RxnMode[mode_name]

    def test_init_with_mode_enum(self, rxn_mode):
        """
        Test that the CondensedGraphOfReactionFeaturizer can be initialized with a RxnMode.
        """
        cgr_featurizer = CondensedGraphOfReactionFeaturizer(mode_=rxn_mode)
        assert cgr_featurizer.mode == rxn_mode

    @pytest.mark.parametrize(
        "reac_prod_mols, expected_output",
        zip(rxn_smis, reac_prod_maps),
        indirect=["reac_prod_mols"],
    )
    def test_map_reac_to_prod(self, reac_prod_mols, expected_output):
        """
        Test that the map_reac_to_prod method returns the correct mapping.
        """
        reac, prod = reac_prod_mols
        assert CondensedGraphOfReactionFeaturizer.map_reac_to_prod(reac, prod) == expected_output

    @pytest.mark.parametrize(
        "reac_prod_mols, reac_prod_maps, cgr_featurizer",
        zip(rxn_smis, reac_prod_maps, AVAILABLE_RXN_MODE_NAMES),
        indirect=["reac_prod_mols", "cgr_featurizer"],
    )
    def test_calc_node_feature_matrix_shape(self, reac_prod_mols, reac_prod_maps, cgr_featurizer):
        """
        Test that the calc_node_feature_matrix method returns the correct node feature matrix.
        """
        reac, prod = reac_prod_mols
        ri2pj, pids, rids = reac_prod_maps
        num_nodes, atom_fdim = cgr_featurizer._calc_node_feature_matrix(
            reac, prod, ri2pj, pids, rids
        ).shape
        assert num_nodes == len(ri2pj) + len(pids) + len(rids)
        assert atom_fdim == cgr_featurizer.atom_fdim

    @pytest.mark.parametrize(
        "reac_prod_mols, reac_prod_maps, cgr_featurizer",
        zip(rxn_smis, reac_prod_maps, AVAILABLE_RXN_MODE_NAMES),
        indirect=["reac_prod_mols", "cgr_featurizer"],
    )
    def test_calc_node_feature_matrix_atomic_number_features(
        self, reac_prod_mols, reac_prod_maps, cgr_featurizer
    ):
        """
        Test that the calc_node_feature_matrix method returns the correct feature matrix for the atomic number features.
        """
        reac, prod = reac_prod_mols
        ri2pj, pids, rids = reac_prod_maps
        atom_featurizer = cgr_featurizer.atom_featurizer

        atomic_num_features_expected = np.array(
            [atom_featurizer.num_only(a) for a in reac.GetAtoms()]
            + [atom_featurizer.num_only(prod.GetAtomWithIdx(pid)) for pid in pids]
        )[:, : atom_featurizer.max_atomic_num + 1]
        atomic_num_features = cgr_featurizer._calc_node_feature_matrix(
            reac, prod, ri2pj, pids, rids
        )[:, : atom_featurizer.max_atomic_num + 1]
        np.testing.assert_equal(atomic_num_features, atomic_num_features_expected)

    @pytest.mark.parametrize(
        "reac_prod_mols, reac_prod_maps, cgr_featurizer, expected_bonds",
        zip(
            rxn_smis,
            reac_prod_maps,
            AVAILABLE_RXN_MODE_NAMES[::2] * 2,
            elements_from_get_bond_is_None_imbalanced,
        ),
        indirect=["reac_prod_mols", "cgr_featurizer"],
    )
    def test_get_bonds_imbalanced(
        self, reac_prod_mols, reac_prod_maps, cgr_featurizer, expected_bonds
    ):
        """
        Test that the get_bonds method returns the correct bonds when modes are imbalanced.
        """
        reac, prod = reac_prod_mols
        ri2pj, pids, _ = reac_prod_maps

        for bond_pair, expect_to_be_None in expected_bonds.items():
            bond_reac, bond_prod = cgr_featurizer._get_bonds(
                reac, prod, ri2pj, pids, reac.GetNumAtoms(), *bond_pair
            )
        assert (bond_reac is None) == expect_to_be_None[0]
        assert (bond_prod is None) == expect_to_be_None[1]

    @pytest.mark.parametrize(
        "reac_prod_mols, reac_prod_maps, cgr_featurizer, expected_bonds",
        zip(
            rxn_smis,
            reac_prod_maps,
            AVAILABLE_RXN_MODE_NAMES[1::2] * 2,
            elements_from_get_bond_is_None_balanced,
        ),
        indirect=["reac_prod_mols", "cgr_featurizer"],
    )
    def test_get_bonds_balanced(
        self, reac_prod_mols, reac_prod_maps, cgr_featurizer, expected_bonds
    ):
        """
        Test that the get_bonds method returns the correct bonds when modes are balanced.
        """
        reac, prod = reac_prod_mols
        ri2pj, pids, _ = reac_prod_maps

        for bond_pair, expect_to_be_None in expected_bonds.items():
            bond_reac, bond_prod = cgr_featurizer._get_bonds(
                reac, prod, ri2pj, pids, reac.GetNumAtoms(), *bond_pair
            )
        assert (bond_reac is None) == expect_to_be_None[0]
        assert (bond_prod is None) == expect_to_be_None[1]

    @pytest.mark.parametrize(
        "cgr_featurizer", AVAILABLE_RXN_MODE_NAMES, indirect=["cgr_featurizer"]
    )
    def test_calc_edge_feature_shape(self, bond_reac_prod, cgr_featurizer):
        """
        Test that the calc_edge_feature method returns the correct edge feature.
        """
        assert cgr_featurizer._calc_edge_feature(*bond_reac_prod).shape == (
            len(cgr_featurizer.bond_featurizer) * 2,
        )

    @pytest.mark.parametrize(
        "reac_prod_mols, reac_prod_maps, cgr_featurizer, expected_bonds",
        zip(
            rxn_smis,
            reac_prod_maps,
            AVAILABLE_RXN_MODE_NAMES[1::2] * 2,
            elements_from_get_bond_is_None_balanced,
        ),
        indirect=["reac_prod_mols", "cgr_featurizer"],
    )
    def test_featurize(self, reac_prod_mols, reac_prod_maps, cgr_featurizer, expected_bonds):
        """
        Test that the get_bonds method returns the correct bonds when modes are balanced.
        """
        molgraph = cgr_featurizer(reac_prod_mols)
        ri2pj, pids, rids = reac_prod_maps

        expected_bonds = [
            list(bond_pair)
            for bond_pair, expect_to_be_None in expected_bonds.items()
            if expect_to_be_None != (True, True)
        ]

        assert molgraph.n_atoms == len(ri2pj) + len(pids) + len(rids)
        assert molgraph.n_bonds == len(expected_bonds) * 2
        assert molgraph.V.shape == (molgraph.n_atoms, cgr_featurizer.atom_fdim)
        assert molgraph.E.shape == (molgraph.n_bonds, cgr_featurizer.bond_fdim)
        assert molgraph.b2a == sum(sorted(expected_bonds), [])
        assert molgraph.b2revb == sum([[i + 1, i] for i in range(0, molgraph.n_bonds, 2)], [])
        a2b_expected = [[] for _ in range(molgraph.n_atoms)]
        for a, b in zip(molgraph.b2a, molgraph.b2revb):
            a2b_expected[a].append(b)
        assert molgraph.a2b == a2b_expected
