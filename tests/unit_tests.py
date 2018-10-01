import pytest

def test_load_split_data():
    import sys
    print(sys.path)
    from chemprop.chemprop.dio import load_split_data
    test_fn = "tests\\data\\qm8crop.csv"
    test, train, validate  = load_split_data(test_fn, 0.25, 0.25, 1)
    (test, train, validate)
