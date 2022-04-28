"""Chemprop unit tests for chemprop/uncertainty/uncertainty_evaluator.py"""
from unittest import TestCase

import numpy as np
import pytest

from chemprop.uncertainty.uncertainty_evaluator import (
    UncertaintyEvaluator,
    build_uncertainty_evaluator,
)


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
    build_uncertainty_evaluator("nll", None, "ensemble", "regression", "mse", None)


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


class TestBuildEvaluator(TestCase):
    """
    Tests build_uncertainty_evaluator function.
    """

    def test_supported(self):
        metrics = {
            "regression": ["nll", "miscalibration_area", "ence", "spearman"],
            "classification": ["auc", "prc-auc", "accuracy", "f1", "mcc"],
            "multiclass": ["cross_entropy", "accuracy", "f1", "mcc"],
        }
        for dtype in metrics:
            for met in metrics[dtype]:
                evaluator = build_uncertainty_evaluator(met, None, None, dtype, None, None)
                self.assertIsInstance(evaluator, UncertaintyEvaluator)

    def test_unsupported(self):
        with self.assertRaises(NotImplementedError):
            evaluator = build_uncertainty_evaluator("nll", None, None, "spectra", "sid", None)


class TestNLLRegression(TestCase):
    """
    Tests NLLRegressionEvaluator class
    """

    def test(self):
        evaluator = build_uncertainty_evaluator(
            evaluation_method="nll",
            calibration_method=None,
            uncertainty_method="ensemble",
            dataset_type="regression",
            loss_function="mse",
            calibrator=None,
        )
        preds = [[0]]
        targets = [[0]]
        unc = [[1]]
        nll = evaluator.evaluate(targets, preds, unc)
        self.assertAlmostEqual(0.3989, np.exp(-1 * nll[0]), places=4)


@pytest.mark.parametrize("targets,preds,uncs,likelihood", [([[0]], [[0]], [[1]], [0.3989])])
def test_nll_regression(nll_regression_evaluator, targets, preds, uncs, likelihood):
    nll_calc = np.array(nll_regression_evaluator.evaluate(targets, preds, uncs))
    likelihood_calc = np.exp(-1 * nll_calc)

    np.testing.assert_array_almost_equal(likelihood, likelihood_calc)


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
