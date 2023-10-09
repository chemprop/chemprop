import pytest
from chemprop.v2.cli.utils.utils import column_str_to_int


@pytest.fixture
def example_header_1():
    return ["smiles1", "target1"]


@pytest.fixture
def example_header_2():
    return ["smiles1", "smiles2", "ignore1", "target1", "target2"]


@pytest.fixture
def example_header_when_no_header():
    return [0, 1, 2, 3]


@pytest.fixture
def example_input_str():
    return ["smiles1", "smiles2"]


@pytest.fixture
def example_input_int():
    return ["3", "4"]


def test_input_None(example_header_1):
    assert column_str_to_int(None, example_header_1) is None


def test_input_str(example_input_str, example_header_2):
    assert column_str_to_int(example_input_str, example_header_2) == [0, 1]


def test_input_str(example_input_int, example_header_2):
    assert column_str_to_int(example_input_int, example_header_2) == [3, 4]


def test_column_not_in_header(example_input_str, example_header_1):
    with pytest.raises(ValueError):
        column_str_to_int(example_input_str, example_header_1)
