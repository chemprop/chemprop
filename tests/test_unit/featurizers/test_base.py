import pytest

from chemprop.featurizers.multihot import MultiHotFeaturizer

def test_abc():
    with pytest.raises(TypeError):
        MultiHotFeaturizer()

@pytest.mark.parametrize(
    "xs,x,i_expected",
    [
        ([0], 0, 0),
        ([], 0, -1),
        (list(range(5)), 5, -1),
        (list(range(5)), 1, 1),
        (list(range(10)), 7, 7)
    ]
)
def test_safe_index(x, xs, i_expected):
    assert MultiHotFeaturizer.safe_index(x, xs) == i_expected