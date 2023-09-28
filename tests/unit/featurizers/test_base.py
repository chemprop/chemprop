import pytest

from chemprop.v2.featurizers.multihot import MultiHotFeaturizer


def test_abc():
    with pytest.raises(TypeError):
        MultiHotFeaturizer()


@pytest.mark.parametrize(
    "xs,x,i_expected",
    [
        ([0], 0, 0),
        ([1], 0, 1),
        (list(range(5)), 5, 5),
        (list(range(5)), 1, 1),
        (list(range(10)), 7, 7),
    ],
)
def test_one_hot_index(x, xs, i_expected):
    assert MultiHotFeaturizer.one_hot_index(x, xs)[0] == i_expected
