import pytest

from chemprop.v2.featurizers.reaction import RxnMode, CondensedGraphOfReactionFeaturizer
from chemprop.v2.utils import make_mol


AVAILABLE_RXN_MODE_NAMES \
    = ['REAC_PROD',
       'REAC_PROD_BALANCE',
       'REAC_DIFF',
       'REAC_DIFF_BALANCE',
       'PROD_DIFF',
       'PROD_DIFF_BALANCE',]


@pytest.fixture
def available_rxn_mode_names():
    return AVAILABLE_RXN_MODE_NAMES


@pytest.fixture(
    params=AVAILABLE_RXN_MODE_NAMES
)
def mode_name(request):
    return request.param


@pytest.fixture
def rxn_mode(mode_name):
    return getattr(RxnMode, mode_name)


@pytest.fixture
def invalid_mode_name():
    return 'INVALID_RXN_MODE'


rxn_smis = [
    # reactant and product with the same number of atoms
    '[CH3:1][H:2]>>[CH3:1].[H:2]',  # reactant and product are balanced and mapped
    '[CH3:2][H:1]>>[H:1].[CH3:2]',  # reactant and product are balanced, mapped but with different atom index order
    '[CH3:1][H]>> [CH3:1].[H:2]',  # reactant and product are balanced and but reactant has less atom-mapped atoms
    '[CH3:1][H:2]>>[H].[CH3:1]',  # reactant and product are balanced and but product has less atom-mapped atoms

    # reactant and product has different numbers of atoms
    '[CH4:1]>>[CH2:1].[H:2][H:3]',  # product has more atoms and more atom-mapped atoms
    '[H:1].[CH2:2][H:3]>>[CH3:2][H:3]',  # reactant with more atoms and atom-mapped atoms.
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


@pytest.fixture
def reac_prod_mols(request):
    return tuple(make_mol(smi, keep_h=True, add_h=False) for smi in request.param.split('>>'))


class TestRxnMode:

    def test_len(self, available_rxn_mode_names):
        """
        Test that the RxnMode class has the correct length.
        """
        assert len(RxnMode) == len(available_rxn_mode_names)

    def test_iteration(self, available_rxn_mode_names):
        """
        Test that the RxnMode class can be iterated over.
        """
        for avail_mode_name, mode in zip(available_rxn_mode_names, RxnMode):
            assert mode.name == avail_mode_name
            assert mode.value == avail_mode_name.lower()

    def test_getitem(self, mode_name):
        """
        Test that the RxnMode class can be indexed with uppercase mode.
        """
        assert RxnMode[mode_name] == getattr(RxnMode, mode_name)
        assert RxnMode[mode_name].name == mode_name
        assert RxnMode[mode_name].value == mode_name.lower()

    def test_getitem_invalid_mode(self, invalid_mode_name):
        """
        Test that the RxnMode class raises a ValueError when indexed with an invalid mode.
        """
        with pytest.raises(KeyError):
            RxnMode[invalid_mode_name]

    def test_get_function(self, mode_name):
        """
        Test that the get function returns the correct RxnMode.
        """
        assert RxnMode.get(mode_name) == getattr(RxnMode, mode_name)
        assert RxnMode.get(mode_name).name == mode_name
        assert RxnMode.get(mode_name).value == mode_name.lower()

    def test_get_function_lowercase_mode(self, mode_name):
        """
        Test that the get function returns the correct RxnMode when given a lowercase mode.
        """
        assert RxnMode.get(mode_name.lower()) == getattr(RxnMode, mode_name)
        assert RxnMode.get(mode_name.lower()).name == mode_name
        assert RxnMode.get(mode_name.lower()).value == mode_name.lower()

    def test_get_function_enum(self, rxn_mode):
        """
        Test that the get function returns the correct RxnMode when given a RxnMode.
        """
        assert RxnMode.get(rxn_mode) == rxn_mode
        assert RxnMode.get(rxn_mode).name == rxn_mode.name
        assert RxnMode.get(rxn_mode).value == rxn_mode.value

    def test_get_function_invalid_mode(self, invalid_mode_name):
        """
        Test that the get function raises a ValueError when given an invalid mode.
        """
        with pytest.raises(ValueError):
            RxnMode.get(invalid_mode_name)

    def test_keys(self, available_rxn_mode_names):
        """
        Test that the keys function returns the correct set of modes.
        """
        assert RxnMode.keys() == set(available_mode.lower() for available_mode in available_rxn_mode_names)


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

    def test_init_with_invalid_mode(self, invalid_mode_name):
        """
        Test that the CondensedGraphOfReactionFeaturizer raises a ValueError when initialized with an invalid mode.
        """
        with pytest.raises(ValueError):
            CondensedGraphOfReactionFeaturizer(mode_=invalid_mode_name)

    @pytest.mark.parametrize("reac_prod_mols, expected_output",
                             zip(rxn_smis, reac_prod_maps),
                             indirect=['reac_prod_mols'])
    def test_map_reac_to_prod(self, reac_prod_mols, expected_output):
        """
        Test that the map_reac_to_prod method returns the correct mapping.
        """
        reac, prod = reac_prod_mols
        assert CondensedGraphOfReactionFeaturizer.map_reac_to_prod(reac, prod) == expected_output
