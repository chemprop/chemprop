import pytest
import torch

from chemprop.uncertainty.evaluator import (  # CalibrationAreaEvaluator,; ConformalMulticlassEvaluator,; ConformalMultilabelEvaluator,; ConformalRegressionEvaluator,; ExpectedNormalizedErrorEvaluator,; MetricEvaluator,
    NLLClassEvaluator,
    NLLMultiEvaluator,
    NLLRegressionEvaluator,
    SpearmanEvaluator,
)


@pytest.mark.parametrize(
    "preds,uncs,targets,mask,likelihood",
    [
        (
            None,
            torch.tensor([[0.8]]),
            torch.ones([1, 1]),
            torch.ones([1, 1], dtype=bool),
            torch.tensor([0.8]),
        ),
        (
            None,
            torch.tensor([[0.8]]),
            torch.zeros([1, 1]),
            torch.ones([1, 1], dtype=bool),
            torch.tensor([0.2]),
        ),
    ],
)
def test_NLLClassEvaluator(preds, uncs, targets, mask, likelihood):
    """
    Testing the NLLClassEvaluator
    """
    evaluator = NLLClassEvaluator()
    nll_calc = evaluator.evaluate(preds, uncs, targets, mask)
    likelihood_calc = torch.exp(-1 * nll_calc)
    torch.testing.assert_close(likelihood_calc, likelihood)


@pytest.mark.parametrize(
    "preds,uncs,targets,mask,likelihood",
    [
        (
            None,
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
            None,
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
def test_NLLMultiEvaluator(preds, uncs, targets, mask, likelihood):
    """
    Testing the NLLMultiEvaluator
    """
    evaluator = NLLMultiEvaluator()
    nll_calc = evaluator.evaluate(preds, uncs, targets, mask)
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


# # Example
# @pytest.mark.parametrize(
#     "targets,preds,uncs,mask,coverage_exp",
#     [
#         (
#             torch.arange(100, 0, -1).view(100, 1),
#             torch.full((100, 1), 70),
#             torch.arange(1, 101).view(100, 1),
#             torch.full((1, 100), True, dtype=torch.bool),
#             torch.tensor([0.7]),
#         ),
#         (
#             torch.tensor([[0, 0.3, 1]]),
#             torch.tensor([[0.2, 0.3, 0.4]]),
#             torch.tensor([[0.5, 0.5, 0.5]]),
#             torch.full((3, 1), True, dtype=torch.bool),
#             torch.tensor([0, 1, 0]),
#         ),
#     ],
# )
# def test_ConformalRegressionEvaluator(preds, uncs, targets, mask, coverage_exp):
#     """
#     Testing the ConformalRegressionEvaluator
#     """
#     evaluator = ConformalRegressionEvaluator()
#     coverage = evaluator.evaluate(preds, uncs, targets, mask)

#     torch.testing.assert_close(coverage, coverage_exp, decimal=3)
