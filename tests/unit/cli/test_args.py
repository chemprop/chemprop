from argparse import ArgumentTypeError

import pytest

from chemprop.cli.utils.args import bool_arg


@pytest.mark.parametrize("value, expected", [("true", True), ("False", False), ("TRUE", True)])
def test_bool_arg(value, expected):
    assert bool_arg(value) is expected


def test_bool_arg_invalid():
    with pytest.raises(ArgumentTypeError):
        bool_arg("yes")
