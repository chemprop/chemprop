import pytest

from chemprop.train.metrics import recall_metric,precision_metric,balanced_accuracy_metric,f1_metric,mcc_metric,prc_auc,accuracy
test_cases = [
    ([0, 1, 1, 0], [0.1, 0.9, 0.8, 0.2]),
    ([1, 1, 0, 0], [0.7, 0.8, 0.3, 0.2]),
    ([0, 0, 1, 1], [0.2, 0.4, 0.6, 0.8]),
    ([1, 0, 0, 1], [0.9, 0.1, 0.3, 0.7]),
    ([0, 1, 0, 1], [0.3, 0.6, 0.4, 0.5])
]

# Expected values
expected_auc = [1.0, 1.0, 1.0, 1.0, 1.0]
expected_prc_auc = [1.0, 1.0, 1.0, 1.0, 1.0]
expected_recall = [1.0, 1.0, 1.0, 1.0, 0.5]
expected_precision = [1.0, 1.0, 1.0, 1.0, 1.0]
expected_balanced_accuracy = [1.0, 1.0, 1.0, 1.0, 0.75]
expected_mcc = [1.0, 1.0, 1.0, 1.0, 0.5773502691896258]
expected_f1 = [1.0, 1.0, 1.0, 1.0, 0.6666666666666666]
expected_accuracy = [1.0, 1.0, 1.0, 1.0, 0.75]
@pytest.mark.parametrize("case, expected", zip(test_cases, expected_prc_auc))
def test_prc_auc(case, expected):
    targets, preds = case
    assert abs(prc_auc(targets, preds) - expected) < 1e-3

@pytest.mark.parametrize("case, expected", zip(test_cases, expected_accuracy))
def test_accuracy(case, expected):
    targets, preds = case
    assert abs(accuracy(targets, preds) - expected) < 1e-3

@pytest.mark.parametrize("case, expected", zip(test_cases, expected_recall))
def test_recall(case, expected):
    targets, preds = case
    assert abs(recall_metric(targets, preds) - expected) < 1e-3

@pytest.mark.parametrize("case, expected", zip(test_cases, expected_precision))
def test_precision(case, expected):
    targets, preds = case
    assert abs(precision_metric(targets, preds) - expected) < 1e-3

@pytest.mark.parametrize("case, expected", zip(test_cases, expected_balanced_accuracy))
def test_balanced_accuracy(case, expected):
    targets, preds = case
    assert abs(balanced_accuracy_metric(targets, preds) - expected) < 1e-3

@pytest.mark.parametrize("case, expected", zip(test_cases, expected_f1))
def test_f1(case, expected):
    targets, preds = case
    assert abs(f1_metric(targets, preds) - expected) < 1e-3

@pytest.mark.parametrize("case, expected", zip(test_cases, expected_mcc))
def test_mcc(case, expected):
    targets, preds = case
    assert abs(mcc_metric(targets, preds) - expected) < 1e-3
