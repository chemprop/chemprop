import pytest
import torch

from chemprop.uncertainty.evaluator import (
    MulticlassConformalEvaluator,
    MultilabelConformalEvaluator,
    RegressionConformalEvaluator,
)


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
            torch.arange(100).unsqueeze(1),
            torch.arange(10, 110).unsqueeze(1),
            torch.ones([100, 1], dtype=torch.bool),
            torch.tensor([0.8]),
        ),
        (
            torch.tensor([[0, 0.3, 1]]),
            torch.tensor([[0.4, 0.6, 0.8]]),
            torch.tensor([[0.5, 0.5, 0.5]]),
            torch.ones([1, 3], dtype=torch.bool),
            torch.tensor([0.0, 1.0, 0.0]),
        ),
        (
            torch.arange(100, 0, -1).unsqueeze(1),
            torch.full((100, 1), 140),
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
