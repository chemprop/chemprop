"""Chemprop unit tests for chemprop/uncertainty/uncertainty_evaluator.py"""
import uuid

import numpy as np
import pytest

from chemprop.uncertainty.uncertainty_evaluator import build_uncertainty_evaluator


@pytest.fixture(params=["regression", "classification", "multiclass"])
def dataset_type(request):
    return request.param


@pytest.fixture(params=["nll", "miscalibration_area", "ence", "spearman"])
def regression_metric(request):
    return request.param


@pytest.fixture(params=["auc", "prc-auc", "accuracy", "f1", "mcc"])
def classification_metric(request):
    return request.param


@pytest.fixture(params=["cross_entropy", "accuracy", "f1", "mcc"])
def multiclass_metric(request):
    return request.param


@pytest.fixture
def nll_regression_evaluator():
    return build_uncertainty_evaluator("nll", None, "ensemble", "regression", "mse", None)


@pytest.fixture
def nll_classification_evaluator():
    return build_uncertainty_evaluator(
        "nll", None, "classification", "classification", "binary_cross_entropy", None
    )


@pytest.fixture
def miscal_regression_evaluator():
    return build_uncertainty_evaluator(
        "miscalibration_area", None, "ensemble", "regression", "mse", None
    )


@pytest.fixture
def spearman_evaluator():
    return build_uncertainty_evaluator("spearman", None, "ensemble", "regression", "mse", None)


def test_build_regression_metric(regression_metric):
    assert build_uncertainty_evaluator(regression_metric, None, None, "regression", None, None)


def test_build_classification_metric(classification_metric):
    assert build_uncertainty_evaluator(
        classification_metric, None, None, "classification", None, None
    )


def test_build_multiclass_metric(multiclass_metric):
    assert build_uncertainty_evaluator(multiclass_metric, None, None, "multiclass", None, None)


@pytest.mark.parametrize("metric", [str(uuid.uuid4()) for _ in range(3)])
def test_build_unsupported_metrics(metric, dataset_type):
    with pytest.raises(NotImplementedError):
        build_uncertainty_evaluator(metric, None, None, "spectra", dataset_type, None)


@pytest.mark.parametrize("targets,preds,uncs,likelihood", [([[0]], [[0]], [[1]], [0.3989])])
def test_nll_regression(nll_regression_evaluator, targets, preds, uncs, likelihood):
    nll_calc = np.array(nll_regression_evaluator.evaluate(targets, preds, uncs))
    likelihood_calc = np.exp(-1 * nll_calc)

    np.testing.assert_array_almost_equal(likelihood, likelihood_calc, decimal=4)


@pytest.mark.parametrize(
    "targets,preds,uncs,likelihood",
    [([[1]], [[0.8]], [[0.8]], [0.8]), ([[0]], [[0.8]], [[0.8]], [0.2])],
)
def test_nll_classificiation(nll_classification_evaluator, targets, preds, uncs, likelihood):
    nll_calc = np.array(nll_classification_evaluator.evaluate(targets, preds, uncs))
    likelihood_calc = np.exp(-1 * nll_calc)

    np.testing.assert_array_almost_equal(likelihood, likelihood_calc)


@pytest.mark.parametrize(
    "targets,preds,uncs,area_exp",
    [
        (np.zeros((100, 1)), np.zeros((100, 1)), np.ones((100, 1)), [0.495]),
        (np.full((100, 1), 100), np.zeros((100, 1)), np.ones((100, 1)), [0.495]),
    ],
)
def test_miscal_regression(miscal_regression_evaluator, targets, preds, uncs, area_exp):
    area = miscal_regression_evaluator.evaluate(targets, preds, uncs)

    np.testing.assert_array_almost_equal(area, area_exp)


@pytest.mark.parametrize(
    "targets,preds,uncs,spearman_exp",
    [
        (
            np.arange(1, 101).reshape((100, 1)),
            np.zeros((100, 1)),
            np.arange(1, 101).reshape((100, 1)),
            [1],
        ),
        (
            np.arange(1, 101).reshape((100, 1)),
            np.zeros((100, 1)),
            -np.arange(1, 101).reshape((100, 1)),
            [-1],
        ),
    ],
)
def test_spearman_regression(spearman_evaluator, targets, preds, uncs, spearman_exp):
    area = spearman_evaluator.evaluate(targets, preds, uncs)

    np.testing.assert_array_almost_equal(area, spearman_exp)
