"""Chemprop unit tests for chemprop/uncertainty/uncertainty_evaluator.py"""
import uuid

import numpy as np
import pytest

from chemprop.uncertainty.uncertainty_evaluator import build_uncertainty_evaluator


# Fixtures
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
    return build_uncertainty_evaluator("nll", None, "ensemble", "regression", "mse", None, False)


@pytest.fixture
def nll_classification_evaluator():
    return build_uncertainty_evaluator(
        "nll", None, "classification", "classification", "binary_cross_entropy", None, False
    )


@pytest.fixture
def miscal_regression_evaluator():
    return build_uncertainty_evaluator(
        "miscalibration_area", None, "ensemble", "regression", "mse", None, False
    )


@pytest.fixture
def spearman_evaluator():
    return build_uncertainty_evaluator("spearman", None, "ensemble", "regression", "mse", None, False)

@pytest.fixture
def conformal_coverage_regression_evaluator():
    return build_uncertainty_evaluator("conformal_coverage", None, None, "regression", "mse", None, False)


@pytest.fixture
def conformal_coverage_multiclass_evaluator():
    return build_uncertainty_evaluator(
        "conformal_coverage", None, None, "multiclass", "cross_entropy", None, False
    )


@pytest.fixture
def conformal_coverage_multilabel_evaluator():
    return build_uncertainty_evaluator(
        "conformal_coverage", None, None, "classification", "auc", None, False
    )


# Tests
def test_build_regression_metric(regression_metric):
    """
    Tests the build_uncertainty_evaluator function's acceptance of the different regression evaluators.
    """
    assert build_uncertainty_evaluator(regression_metric, None, None, "regression", None, None, False)


def test_build_classification_metric(classification_metric):
    """
    Tests the build_uncertainty_evaluator function's acceptance of the different classification evaluators.
    """
    assert build_uncertainty_evaluator(
        classification_metric, None, None, "classification", None, None, False
    )


def test_build_multiclass_metric(multiclass_metric):
    """
    Tests the build_uncertainty_evaluator function's acceptance of the different multiclass evaluators.
    """
    assert build_uncertainty_evaluator(multiclass_metric, None, None, "multiclass", None, None, False)


@pytest.mark.parametrize("metric", [str(uuid.uuid4()) for _ in range(3)])
def test_build_unsupported_metrics(metric, dataset_type):
    """
    Tests build_uncertainty_evaluator function's unsupported error for unknown metric strings.
    """
    with pytest.raises(NotImplementedError):
        build_uncertainty_evaluator(metric, None, None, dataset_type, None, None, False)


@pytest.mark.parametrize("targets,preds,uncs,mask,likelihood", [([[0]], [[0]], [[1]], [[True]], [0.3989])])
def test_nll_regression(nll_regression_evaluator, targets, preds, uncs, mask, likelihood):
    """
    Tests the result of the NLL regression UncertaintyEvaluator.
    """
    nll_calc = np.array(nll_regression_evaluator.evaluate(targets, preds, uncs, mask))
    likelihood_calc = np.exp(-1 * nll_calc)

    np.testing.assert_array_almost_equal(likelihood, likelihood_calc, decimal=4)


@pytest.mark.parametrize(
    "targets,preds,uncs,mask,likelihood",
    [([[1]], [[0.8]], [[0.8]], [[True]], [0.8]), ([[0]], [[0.8]], [[0.8]], [[True]], [0.2])],
)
def test_nll_classificiation(nll_classification_evaluator, targets, preds, uncs, mask, likelihood):
    """
    Tests the result of the NLL classification UncertaintyEvaluator.
    """
    nll_calc = np.array(nll_classification_evaluator.evaluate(targets, preds, uncs, mask))
    likelihood_calc = np.exp(-1 * nll_calc)

    np.testing.assert_array_almost_equal(likelihood, likelihood_calc)


@pytest.mark.parametrize(
    "targets,preds,uncs,mask,area_exp",
    [
        (np.zeros((100, 1)), np.zeros((100, 1)), np.ones((100, 1)), np.full((1, 100), True, dtype=bool), [0.495]),
        (np.full((100, 1), 100), np.zeros((100, 1)), np.ones((100, 1)), np.full((1, 100), True, dtype=bool), [0.495]),
    ],
)
def test_miscal_regression(miscal_regression_evaluator, targets, preds, uncs, mask, area_exp):
    """
    Tests the result of the miscalibration_area UncertaintyEvaluator.
    """
    area = miscal_regression_evaluator.evaluate(targets, preds, uncs, mask)

    np.testing.assert_array_almost_equal(area, area_exp)


@pytest.mark.parametrize(
    "targets,preds,uncs,mask,spearman_exp",
    [
        (
            np.arange(1, 101).reshape((100, 1)),
            np.zeros((100, 1)),
            np.arange(1, 101).reshape((100, 1)),
            np.full((1, 100), True, dtype=bool),
            [1],
        ),
        (
            np.arange(1, 101).reshape((100, 1)),
            np.zeros((100, 1)),
            -np.arange(1, 101).reshape((100, 1)),
            np.full((1, 100), True, dtype=bool),
            [-1],
        ),
    ],
)
def test_spearman_regression(spearman_evaluator, targets, preds, uncs, mask, spearman_exp):
    """
    Tests the result of the spearman rank correlation UncertaintyEvaluator.
    """
    area = spearman_evaluator.evaluate(targets, preds, uncs, mask)

    np.testing.assert_array_almost_equal(area, spearman_exp)


@pytest.mark.parametrize(
    "targets,preds,uncs,mask,coverage_exp",
    [
        (
            np.arange(1, 101, 1).reshape(100, 1),
            np.arange(100, 0, -1).reshape(100, 1),
            np.full((100, 1), 70),
            np.full((1, 100), True, dtype=bool),
            [0.7],
        ),
        (
            np.array([[0.5, 0.5, 0.5]]),
            np.array([[0, 0.3, 1]]),
            np.array([[0.2, 0.3, 0.4]]),
            np.full((3, 1), True, dtype=bool),
            [0, 1, 0],
        ),
    ],
)
def test_conformal_regression_coverage(
    conformal_coverage_regression_evaluator, targets, preds, uncs, mask, coverage_exp
):
    """
    Tests the result of the conformal_coverage for regression UncertaintyEvaluator.
    """

    coverage = conformal_coverage_regression_evaluator.evaluate(targets, preds, uncs, mask)

    np.testing.assert_array_almost_equal(coverage, coverage_exp, decimal=3)


@pytest.mark.parametrize(
    "targets,preds,uncs,mask,coverage_exp",
    [
        (
            np.array([0, 1, 1]).reshape(3, 1),
            np.full((3, 1, 2), 0.5),
            np.array([[1, 0], [0, 1], [1, 0]]).reshape(3, 1, 2),
            np.full((1, 3), True, dtype=bool),
            [0.6666],
        ),
        (
            np.array([0, 1, 2]).reshape(3, 1),
            np.full((3, 1, 3), 0.5),
            np.array([[1, 0, 0], [1, 0, 0], [1, 0, 0]]).reshape(3, 1, 3),
            np.full((1, 3), True, dtype=bool),
            [0.3333],
        ),
    ],
)
def test_conformal_multiclass_coverage(
    conformal_coverage_multiclass_evaluator, targets, preds, uncs, mask, coverage_exp
):
    """
    Tests the result of the conformal_coverage for multiclass UncertaintyEvaluator.
    """
    coverage = conformal_coverage_multiclass_evaluator.evaluate(targets, preds, uncs, mask)

    np.testing.assert_array_almost_equal(coverage, coverage_exp, decimal=3)


@pytest.mark.parametrize(
    "targets,preds,uncs,mask,coverage_exp",
    [
        (
            np.array([[0, 0], [1, 0], [1, 1]]),
            np.full((3, 2), 0.5),
            np.array([[0, 0, 0, 0], [0, 1, 1, 1], [0, 0, 0, 0]]),
            np.full((2, 3), True, dtype=bool),
            [0.6666, 0.3333],
        )
    ],
)
def test_conformal_multilabel_coverage(
    conformal_coverage_multilabel_evaluator, targets, preds, uncs, mask, coverage_exp
):
    """
    Tests the result of the conformal_coverage for multilabel UncertaintyEvaluator.
    """
    coverage = conformal_coverage_multilabel_evaluator.evaluate(targets, preds, uncs, mask)

    np.testing.assert_array_almost_equal(coverage, coverage_exp, decimal=3)
