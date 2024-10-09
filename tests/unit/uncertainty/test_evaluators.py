# import pytest
# import torch

# from chemprop.uncertainty.evaluator import (
#     CalibrationAreaEvaluator,
#     ConformalMulticlassEvaluator,
#     ConformalMultilabelEvaluator,
#     ConformalRegressionEvaluator,
#     ExpectedNormalizedErrorEvaluator,
#     MetricEvaluator,
#     NLLClassEvaluator,
#     NLLMultiEvaluator,
#     NLLRegressionEvaluator,
#     SpearmanEvaluator,
# )


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
