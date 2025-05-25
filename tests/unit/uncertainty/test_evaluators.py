import pytest
import torch

from chemprop.uncertainty.evaluator import (
    CalibrationAreaEvaluator,
    ExpectedNormalizedErrorEvaluator,
    MulticlassConformalEvaluator,
    MultilabelConformalEvaluator,
    NLLClassEvaluator,
    NLLMulticlassEvaluator,
    NLLRegressionEvaluator,
    RegressionConformalEvaluator,
    SpearmanEvaluator,
)


@pytest.mark.parametrize(
    "uncs,targets,mask,likelihood",
    [
        (
            torch.tensor([[0.8]]),
            torch.ones([1, 1]),
            torch.ones([1, 1], dtype=bool),
            torch.tensor([0.8]),
        ),
        (
            torch.tensor([[0.8]]),
            torch.zeros([1, 1]),
            torch.ones([1, 1], dtype=bool),
            torch.tensor([0.2]),
        ),
    ],
)
def test_NLLClassEvaluator(uncs, targets, mask, likelihood):
    """
    Testing the NLLClassEvaluator
    """
    evaluator = NLLClassEvaluator()
    nll_calc = evaluator.evaluate(uncs, targets, mask)
    likelihood_calc = torch.exp(-1 * nll_calc)
    torch.testing.assert_close(likelihood_calc, likelihood)


@pytest.mark.parametrize(
    "uncs,targets,mask,likelihood",
    [
        (
            torch.tensor(
                [
                    [[0.29, 0.22, 0.49]],
                    [[0.35, 0.19, 0.46]],
                    [[0.55, 0.38, 0.07]],
                    [[0.15, 0.29, 0.56]],
                    [[0.08, 0.68, 0.24]],
                ]
            ),
            torch.tensor([[0], [2], [2], [0], [1]]),
            torch.ones([5, 1], dtype=bool),
            torch.tensor([0.24875443]),
        ),
        (
            torch.tensor(
                [
                    [[8.7385e-01, 8.3770e-04, 3.3212e-02, 9.2103e-02]],
                    [[7.2274e-03, 1.0541e-01, 8.8703e-01, 3.2886e-04]],
                    [[1.7376e-03, 9.9478e-01, 1.4227e-03, 2.0596e-03]],
                    [[2.6487e-04, 1.3251e-03, 2.4325e-02, 9.7409e-01]],
                ]
            ),
            torch.tensor([[0], [2], [1], [3]]),
            torch.ones([4, 1], dtype=bool),
            torch.tensor([0.93094635]),
        ),
    ],
)
def test_NLLMulticlassEvaluator(uncs, targets, mask, likelihood):
    """
    Testing the NLLMulticlassEvaluator
    """
    evaluator = NLLMulticlassEvaluator()
    nll_calc = evaluator.evaluate(uncs, targets, mask)
    likelihood_calc = torch.exp(-1 * nll_calc)
    torch.testing.assert_close(likelihood_calc, likelihood)


@pytest.mark.parametrize(
    "preds,uncs,targets,mask,likelihood",
    [
        (
            torch.zeros([1, 1]),
            torch.ones([1, 1]),
            torch.zeros([1, 1]),
            torch.ones([1, 1], dtype=bool),
            torch.tensor([0.39894228]),
        ),
        (
            torch.zeros([2, 2]),
            torch.ones([2, 2]),
            torch.zeros([2, 2]),
            torch.ones([2, 2], dtype=bool),
            torch.tensor([0.39894228, 0.39894228]),
        ),
    ],
)
def test_NLLRegressionEvaluator(preds, uncs, targets, mask, likelihood):
    """
    Testing the NLLRegressionEvaluator
    """
    evaluator = NLLRegressionEvaluator()
    nll_calc = evaluator.evaluate(preds, uncs, targets, mask)
    likelihood_calc = torch.exp(-1 * nll_calc)
    torch.testing.assert_close(likelihood_calc, likelihood)


@pytest.mark.parametrize(
    "preds,uncs,targets,mask,spearman_exp",
    [
        (
            torch.zeros(100, 1, dtype=float),
            torch.arange(1, 101, dtype=float).unsqueeze(1),
            torch.arange(1, 101, dtype=float).unsqueeze(1),
            torch.ones(100, 1, dtype=bool),
            torch.tensor([1.0]),
        ),
        (
            torch.zeros(100, 1, dtype=float),
            -torch.arange(1, 101, dtype=float).unsqueeze(1),
            torch.arange(1, 101, dtype=float).unsqueeze(1),
            torch.ones(100, 1, dtype=bool),
            torch.tensor([-1.0]),
        ),
    ],
)
def test_SpearmanEvaluator(preds, uncs, targets, mask, spearman_exp):
    """
    Testing the SpearmanEvaluator
    """
    evaluator = SpearmanEvaluator()
    area = evaluator.evaluate(preds, uncs, targets, mask)
    torch.testing.assert_close(area, spearman_exp)


@pytest.mark.parametrize(
    "uncs,targets,mask,coverage",
    [
        (
            torch.tensor([[[1, 0], [0, 1]], [[0, 1], [1, 0]], [[1, 0], [1, 0]]]),
            torch.tensor([[0, 0], [1, 0], [1, 1]]),
            torch.ones([3, 2], dtype=torch.bool),
            torch.tensor([0.66666, 0.33333]),
        )
    ],
)
def test_MulticlassConformalEvaluator(uncs, targets, mask, coverage):
    """
    Testing the MulticlassConformalEvaluator
    """
    evaluator = MulticlassConformalEvaluator()
    coverage_cal = evaluator.evaluate(uncs, targets, mask)

    torch.testing.assert_close(coverage_cal, coverage)


@pytest.mark.parametrize(
    "uncs,targets,mask,coverage",
    [
        (
            torch.tensor([[0, 0, 0, 0], [0, 1, 1, 1], [0, 0, 0, 0]]),
            torch.tensor([[0, 0], [1, 0], [1, 1]]),
            torch.ones([3, 2], dtype=torch.bool),
            torch.tensor([0.66666, 0.33333]),
        )
    ],
)
def test_MultilabelConformalEvaluator(uncs, targets, mask, coverage):
    """
    Testing the MultilabelConformalEvaluator
    """
    evaluator = MultilabelConformalEvaluator()
    coverage_cal = evaluator.evaluate(uncs, targets, mask)

    torch.testing.assert_close(coverage_cal, coverage)


@pytest.mark.parametrize(
    "preds,uncs,targets,mask,coverage",
    [
        (
            torch.arange(100).unsqueeze(1),
            torch.arange(100).unsqueeze(1) / 2,
            torch.arange(10, 110).unsqueeze(1),
            torch.ones([100, 1], dtype=torch.bool),
            torch.tensor([0.8]),
        ),
        (
            torch.tensor([[0, 0.3, 1]]),
            torch.tensor([[0.2, 0.3, 0.4]]),
            torch.tensor([[0.5, 0.5, 0.5]]),
            torch.ones([1, 3], dtype=torch.bool),
            torch.tensor([0.0, 1.0, 0.0]),
        ),
        (
            torch.arange(100, 0, -1).unsqueeze(1),
            torch.full((100, 1), 70),
            torch.arange(1, 101, 1).unsqueeze(1),
            torch.ones([100, 1], dtype=torch.bool),
            torch.tensor([0.7]),
        ),
    ],
)
def test_RegressionConformalEvaluator(preds, uncs, targets, mask, coverage):
    """
    Testing the RegressionConformalEvaluator
    """
    evaluator = RegressionConformalEvaluator()
    coverage_cal = evaluator.evaluate(preds, uncs, targets, mask)

    torch.testing.assert_close(coverage_cal, coverage)


@pytest.mark.parametrize(
    "preds,uncs,targets,mask,miscal_area",
    [
        (
            torch.zeros(100).unsqueeze(1),
            torch.ones(100).unsqueeze(1),
            torch.zeros(100).unsqueeze(1),
            torch.ones([100, 1], dtype=torch.bool),
            torch.tensor([0.495]),
        ),
        (
            torch.ones(100).unsqueeze(1),
            torch.ones(100).unsqueeze(1),
            torch.ones(100, 1) * 100,
            torch.ones([100, 1], dtype=torch.bool),
            torch.tensor([0.495]),
        ),
    ],
)
def test_CalibrationAreaEvaluator(preds, uncs, targets, mask, miscal_area):
    """
    Testing the CalibrationAreaEvaluator
    """
    evaluator = CalibrationAreaEvaluator()
    miscal_area_cal = evaluator.evaluate(preds, uncs, targets, mask)

    torch.testing.assert_close(miscal_area_cal, miscal_area)


@pytest.mark.parametrize(
    "preds,uncs,targets,mask,ence",
    [
        (
            torch.zeros(100, 1),
            torch.ones(100, 1),
            torch.zeros(100, 1),
            torch.ones([100, 1], dtype=torch.bool),
            torch.tensor([1.0]),
        ),
        (
            torch.linspace(1, 100, steps=100).unsqueeze(1),
            torch.linspace(1, 10, steps=100).unsqueeze(1),
            torch.linspace(1, 100, steps=100).unsqueeze(1)
            + torch.tensor([-2, -1, 1, 2]).repeat(25).unsqueeze(1),
            torch.ones([100, 1], dtype=torch.bool),
            torch.tensor([0.392]),
        ),
    ],
)
def test_ExpectedNormalizedErrorEvaluator(preds, uncs, targets, mask, ence):
    """
    Testing the ExpectedNormalizedErrorEvaluator
    """
    evaluator = ExpectedNormalizedErrorEvaluator()
    ence_cal = evaluator.evaluate(preds, uncs, targets, mask)

    torch.testing.assert_close(ence_cal, ence)
