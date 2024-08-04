# flake8: noqa
import numpy as np
import pytest
from rdkit import Chem

from chemprop.featurizers import (
    MorganBinaryFeaturizer,
    MorganCountFeaturizer,
    RDKit2DFeaturizer,
    V1RDKit2DFeaturizer,
)
from chemprop.featurizers.molecule import NO_DESCRIPTASTORUS


@pytest.fixture
def mol():
    return Chem.MolFromSmiles("Fc1cccc(C2(c3nnc(Cc4cccc5ccccc45)o3)CCOCC2)c1")


# fmt: off
@pytest.fixture
def morgan_binary_bits():
    return np.array([[  80,  230,  332,  378,  429,  450,  502,  503,  523,  544,  556,
                      645,  649,  656,  663,  699,  772,  875,  917,  926,  950, 1039,
                     1060, 1087, 1088, 1104, 1136, 1162, 1164, 1199, 1349, 1357, 1380,
                     1405, 1430, 1487, 1510, 1561, 1573, 1597, 1604, 1670, 1742, 1747,
                     1750, 1824, 1855, 1873, 1928]])


@pytest.fixture
def morgan_count_bits():
    return np.array([ 1,  1,  1,  2,  1,  1,  1,  1,  1,  1,  1,  2,  1,  2,  1,  1,  1,
                      1,  1,  4,  2,  2,  1,  2,  4,  1,  1,  2,  2,  2,  1,  1,  7,  1,
                      1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  6,  2,  1, 11,  1])


@pytest.fixture
def morgan_binary_custom():
    return np.array([[ 15,  36,  49,  63,  64,  80, 112, 138, 140, 175, 230, 275, 301,
                     325, 332, 333, 339, 356, 378, 381, 406, 429, 450, 463, 465, 478,
                     486, 502, 503, 517, 523, 524, 537, 544, 549, 554, 556, 573, 579,
                     580, 645, 646, 647, 649, 652, 656, 663, 699, 718, 721, 723, 726,
                     731, 772, 773, 800, 818, 821, 828, 831, 836, 849, 865, 875, 887,
                     894, 904, 917, 926, 950, 951, 989]])


@pytest.fixture
def rdkit_2d_values():
    return np.array([     13.9511,      13.9511,       0.2603,      -0.5096,
                           0.4909,      16.1724,     388.442 ,     367.274 ,
                         388.1587,     146.    ,       0.    ,       0.2267,
                          -0.4239,       0.4239,       0.2267,       0.8966,
                           1.6897,       2.5517,      19.1421,       9.7377,
                           2.4117,      -2.34  ,       2.4051,      -2.3511,
                           5.8532,       0.054 ,       3.2361,       1.5168,
                        1143.0568,      19.6836,      15.9753,      15.9753,
                          14.244 ,       9.8787,       9.8787,       7.5208,
                           7.5208,       5.8214,       5.8214,       4.26  ,
                           4.26  ,      -3.05  , 9626644.372 ,      18.0088,
                           7.4091,       3.3162,     167.8922,       9.154 ,
                           5.8172,       0.    ,      11.7814,       0.    ,
                           0.    ,       0.    ,       4.3904,       0.    ,
                          10.1974,      54.5973,      46.8737,      13.2138,
                          11.8358,      13.5444,      10.7724,       0.    ,
                          10.1974,       0.    ,      24.6775,      13.2138,
                          95.4556,       0.    ,       0.    ,       0.    ,
                           4.3904,       0.    ,       0.    ,      23.4111,
                          16.5727,       5.8172,      35.75  ,      71.1472,
                           0.    ,      10.7724,       0.    ,      48.15  ,
                           5.415 ,       4.3904,       0.    ,       5.8172,
                          44.2577,      11.1269,      16.8388,      12.1327,
                          24.2655,      34.4628,       9.154 ,      25.6895,
                           0.    ,       0.    ,      11.1016,       1.4962,
                           0.851 ,      21.1832,       1.9333,       1.1618,
                           0.    ,       0.25  ,      29.    ,       0.    ,
                           4.    ,       0.    ,       1.    ,       1.    ,
                           3.    ,       1.    ,       4.    ,       4.    ,
                           0.    ,       5.    ,       4.    ,       0.    ,
                           1.    ,       1.    ,       5.    ,       5.0492,
                         108.285 ,       0.    ,       0.    ,       0.    ,
                           0.    ,       0.    ,       2.    ,       0.    ,
                           0.    ,       0.    ,       0.    ,       0.    ,
                           0.    ,       0.    ,       0.    ,       0.    ,
                           2.    ,       0.    ,       0.    ,       0.    ,
                           0.    ,       0.    ,       0.    ,       0.    ,
                           0.    ,       0.    ,       0.    ,       0.    ,
                           0.    ,       0.    ,       0.    ,       0.    ,
                           0.    ,       0.    ,       0.    ,       3.    ,
                           0.    ,       1.    ,       0.    ,       0.    ,
                           0.    ,       0.    ,       1.    ,       0.    ,
                           0.    ,       1.    ,       0.    ,       0.    ,
                           0.    ,       0.    ,       0.    ,       0.    ,
                           0.    ,       0.    ,       0.    ,       0.    ,
                           0.    ,       0.    ,       0.    ,       0.    ,
                           0.    ,       0.    ,       0.    ,       0.    ,
                           0.    ,       0.    ,       0.    ,       0.    ,
                           0.    ,       0.    ,       0.    ,       0.    ,
                           0.    ,       0.    ,       0.    ,       0.    ,
                           0.    ,       0.    ,       0.    ,       0.    ,
                           0.    ,       0.    ,       0.    ,       0.    ,
                           0.    ,       0.    ])

@pytest.fixture
def v1_rdkit_2d_values():  
    return np.array([      1.5168,    1143.0568,      19.6836,      15.9753,
                          15.9753,      14.244 ,       9.8787,       9.8787,
                           7.5208,       7.5208,       5.8214,       5.8214,
                           4.26  ,       4.26  ,       5.415 ,       4.3904,
                           0.    ,       5.8172,      44.2577,      11.1269,
                          16.8388,      12.1327,      24.2655,      34.4628,
                           9.154 ,     388.1587,       0.8966,       1.6897,
                           2.5517,       0.25  ,      -3.05  ,      29.    ,
                         367.274 , 9626644.372 ,      18.0088,       7.4091,
                           3.3162,     167.8922,      13.9511,       0.4239,
                          13.9511,       0.2267,       0.2603,       0.2267,
                          -0.5096,      -0.4239,       5.0492,     108.285 ,
                         388.442 ,       0.    ,       4.    ,       0.    ,
                           1.    ,       1.    ,       3.    ,       1.    ,
                           4.    ,       4.    ,       0.    ,       5.    ,
                           0.    ,       4.    ,       0.    ,       1.    ,
                           1.    ,     146.    ,       9.154 ,       5.8172,
                           0.    ,      11.7814,       0.    ,       0.    ,
                           0.    ,       4.3904,       0.    ,      10.1974,
                          54.5973,      46.8737,      13.2138,      11.8358,
                           5.    ,      13.5444,      10.7724,       0.    ,
                          10.1974,       0.    ,      24.6775,      13.2138,
                          95.4556,       0.    ,       0.    ,       0.    ,
                           4.3904,       0.    ,       0.    ,      23.4111,
                          16.5727,       5.8172,      35.75  ,      71.1472,
                           0.    ,      10.7724,       0.    ,      48.15  ,
                          25.6895,       0.    ,       0.    ,      11.1016,
                           1.4962,       0.851 ,      21.1832,       1.9333,
                           1.1618,       0.    ,       0.    ,       0.    ,
                           0.    ,       0.    ,       0.    ,       2.    ,
                           0.    ,       0.    ,       0.    ,       0.    ,
                           0.    ,       0.    ,       0.    ,       0.    ,
                           0.    ,       2.    ,       0.    ,       0.    ,
                           0.    ,       0.    ,       0.    ,       0.    ,
                           0.    ,       0.    ,       0.    ,       0.    ,
                           0.    ,       0.    ,       0.    ,       0.    ,
                           0.    ,       0.    ,       0.    ,       0.    ,
                           3.    ,       0.    ,       1.    ,       0.    ,
                           0.    ,       0.    ,       0.    ,       1.    ,
                           0.    ,       0.    ,       1.    ,       0.    ,
                           0.    ,       0.    ,       0.    ,       0.    ,
                           0.    ,       0.    ,       0.    ,       0.    ,
                           0.    ,       0.    ,       0.    ,       0.    ,
                           0.    ,       0.    ,       0.    ,       0.    ,
                           0.    ,       0.    ,       0.    ,       0.    ,
                           0.    ,       0.    ,       0.    ,       0.    ,
                           0.    ,       0.    ,       0.    ,       0.    ,
                           0.    ,       0.    ,       0.    ,       0.    ,
                           0.    ,       0.    ,       0.    ,       0.    ,
                           0.    ,       0.    ,       0.    ,       0.4909])
# fmt: on


def test_morgan_binary(mol, morgan_binary_bits):
    featurizer = MorganBinaryFeaturizer()
    features = featurizer(mol)

    np.testing.assert_array_almost_equal(np.nonzero(features), morgan_binary_bits)


def test_morgan_count(mol, morgan_count_bits, morgan_binary_bits):
    featurizer = MorganCountFeaturizer()
    features = featurizer(mol)

    np.testing.assert_array_almost_equal(features[np.nonzero(features)], morgan_count_bits)


def test_morgan_binary_custom(mol, morgan_binary_custom):
    featurizer = MorganBinaryFeaturizer(radius=3, length=1024)
    features = featurizer(mol)

    np.testing.assert_array_almost_equal(np.nonzero(features), morgan_binary_custom)


def test_rdkit_2d(mol, rdkit_2d_values):
    featurizer = RDKit2DFeaturizer()
    features = featurizer(mol)

    np.testing.assert_array_almost_equal(features, rdkit_2d_values, decimal=2)


@pytest.mark.skipif(NO_DESCRIPTASTORUS, reason="Descriptastorus not installed")
def test_v1_rdkit_2d(mol, v1_rdkit_2d_values):
    featurizer = V1RDKit2DFeaturizer()
    features = featurizer(mol)

    np.testing.assert_array_almost_equal(features, v1_rdkit_2d_values, decimal=2)
