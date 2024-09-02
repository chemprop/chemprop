import pytest
import torch

from chemprop.uncertainty.evaluator import (
    CalibrationAreaEvaluator,
    ExpectedNormalizedErrorEvaluator,
)


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
