"""Chemprop unit tests for chemprop/v2/models/loss.py"""
from types import SimpleNamespace

import numpy as np
import torch
import pytest

from chemprop.v2.models.loss import (
    BoundedMSELoss,
    MVELoss,
    BinaryDirichletLoss,
    EvidentialLoss,
    BCELoss,
)
    
@pytest.mark.parametrize(
    "preds,targets,lt_targets,gt_targets,mse",
    [
        (
            torch.tensor([[-3, 2], [1, -1]], dtype=float),
            torch.zeros([2, 2], dtype=float),
            torch.zeros([2, 2], dtype=bool),
            torch.zeros([2, 2], dtype=bool),
            15
        ),
        (
            torch.tensor([[-3, 2], [1, -1]], dtype=float),
            torch.zeros([2, 2], dtype=float),
            torch.zeros([2, 2], dtype=bool),
            torch.ones([2, 2], dtype=bool),
            10
        ),
        (
            torch.tensor([[-3, 2], [1, -1]], dtype=float),
            torch.zeros([2, 2], dtype=float),
            torch.ones([2, 2], dtype=bool),
            torch.zeros([2, 2], dtype=bool),
            5
        ),
    ]
)

def test_BoundedMSE(preds, targets, lt_targets, gt_targets, mse):
    """
    Testing the bounded_mse loss function
    """
    loss = BoundedMSELoss(preds, targets, lt_targets, gt_targets)
    assert loss.sum() == mse

@pytest.mark.parametrize(
    "preds,targets,likelihood",
    [(
        torch.tensor([[0, 1]], dtype=float),
        torch.zeros([1, 1]),
        [[0.3989]],
    )]
)
def test_MVE(preds, targets, likelihood):
    """
    Tests the normal_mve loss function
    """
    nll_calc = MVELoss(preds, targets)
    likelihood_calc = np.exp(-1 * nll_calc)
    np.testing.assert_array_almost_equal(likelihood, likelihood_calc, decimal=4)

@pytest.mark.parametrize(
    "preds,targets,v_kl,expected_loss",
    [
        (
            torch.tensor([[2, 2]]),
            torch.ones([1, 1]),
            0,
            [[0.6]]
        ),
        (
            torch.tensor([[2, 2]]),
            torch.ones([1, 1]),
            0.2,
            [[0.63862943]]
        )
    ]
)
def test_BinaryDirichlet(preds, targets, v_kl, expected_loss):
    """
    Test on the dirichlet loss function for classification.
    Note these values were not hand derived, just testing for
    dimensional consistency.
    """
    loss = BinaryDirichletLoss(preds, targets, v_kl=v_kl)
    np.testing.assert_array_almost_equal(loss, expected_loss)
    
@pytest.mark.parametrize(
    "preds,targets",
    [
        (
            torch.ones([1, 1]),
            torch.ones([1, 1]),
        ),
    ]
)
def test_BinaryDirichlet_wrong_dimensions(preds, targets):
    """
    Test on the dirichlet loss function for classification
    for dimension errors.
    """
    with pytest.raises(RuntimeError):
        BinaryDirichletLoss(preds, targets)

@pytest.mark.parametrize(
    "preds,targets,v_kl,expected_loss",
    [
        (
            torch.tensor([[2, 2, 2, 2]]),
            torch.ones([1, 1]),
            0,
            [[1.56893861]]
        ),
        (
            torch.tensor([[2, 2, 2, 2]]),
            torch.ones([1, 1]),
            0.2,
            [[2.768938541]]
        )
    ]
)
def test_Evidential(preds, targets, v_kl, expected_loss):
    """
    Test on the evidential loss function for classification.
    Note these values were not hand derived, just testing for
    dimensional consistency.
    """
    loss = EvidentialLoss(preds, targets, v_kl=v_kl)
    np.testing.assert_array_almost_equal(loss, expected_loss)


@pytest.mark.parametrize(
    "preds,targets",
    [
        (
            torch.ones([2, 2]),
            torch.ones([2, 2]),
        ),
    ]
)
def test_Evidential_wrong_dimensions(preds, targets):
    """
    Test on the dirichlet loss function for classification
    for dimension errors.
    """
    with pytest.raises(RuntimeError):
        EvidentialLoss(preds, targets)

@pytest.mark.parametrize(
    "preds,targets,expected_loss",
    [
        (
            torch.tensor([2, 2]),
            torch.ones([1, 1]),
            [[0.1269]]
        ),
        (
            torch.tensor([0.5, 0.5]),
            torch.ones([1, 1]),
            [[0.4741]]
        )
    ]
)
def test_BCE(preds, targets):
    """
    Test on the evidential loss function for classification.
    Note these values were not hand derived, just testing for
    dimensional consistency.
    """
    loss = BCELoss(preds, targets)
    np.testing.assert_array_almost_equal(loss, expected_loss)
