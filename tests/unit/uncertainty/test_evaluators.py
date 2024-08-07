import pytest
import torch

from chemprop.uncertainty.evaluator import (
    # CalibrationAreaEvaluator,
    # ConformalMulticlassEvaluator,
    # ConformalMultilabelEvaluator,
    # ConformalRegressionEvaluator,
    # ExpectedNormalizedErrorEvaluator,
    # MetricEvaluator,
    NLLClassEvaluator,
    # NLLMultiEvaluator,
    NLLRegressionEvaluator,
    SpearmanEvaluator,
)

@pytest.mark.parametrize(
    "preds,uncs,targets,mask,likelihood",
    [
       (
        torch.tensor([[0.8]]),
        torch.tensor([[0.8]]),
        torch.ones([1, 1]),
        torch.ones([1, 1], dtype = bool),
        torch.tensor([0.8]),
       ),
       (
        torch.tensor([[0.8]]),
        torch.tensor([[0.8]]),
        torch.zeros([1, 1]),
        torch.ones([1, 1], dtype = bool),
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
     likelihood_calc = torch.exp(-1* nll_calc)
     torch.testing.assert_close(likelihood_calc, likelihood)

@pytest.mark.parametrize(
    "preds,uncs,targets,mask,likelihood",
    [
       (
        torch.zeros([1, 1]),
        torch.ones([1, 1]),
        torch.zeros([1, 1]),
        torch.ones([1, 1], dtype = bool),
        torch.tensor([0.39894228]),
       ),
        (
        torch.zeros([2, 2]),
        torch.ones([2, 2]),
        torch.zeros([2, 2]),
        torch.ones([2, 2], dtype = bool),
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
     likelihood_calc = torch.exp(-1* nll_calc)
     torch.testing.assert_close(likelihood_calc, likelihood)

@pytest.mark.parametrize(
    "preds,uncs,targets,mask,spearman_exp",
    [
       (
        torch.zeros(100, 1, dtype = float),
        torch.arange(1, 101, dtype = float).unsqueeze(1),
        torch.arange(1, 101, dtype = float).unsqueeze(1),
        torch.ones(100, 1, dtype = bool),
        torch.tensor([1.]),
       ),
        (
        torch.zeros(100, 1, dtype = float),
        -torch.arange(1, 101, dtype = float).unsqueeze(1),
        torch.arange(1, 101, dtype = float).unsqueeze(1),
        torch.ones(100, 1, dtype = bool) ,
        torch.tensor([-1.]),
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
