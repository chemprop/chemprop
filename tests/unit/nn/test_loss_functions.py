"""Chemprop unit tests for chemprop/models/loss.py"""

import numpy as np
import pytest
import torch

from chemprop.nn.loss import (
    BCELoss,
    BinaryDirichletLoss,
    BinaryMCCLoss,
    BoundedMSELoss,
    CrossEntropyLoss,
    EvidentialLoss,
    MulticlassDirichletLoss,
    MulticlassMCCLoss,
    MVELoss,
    SIDLoss,
    WassersteinLoss,
)


@pytest.mark.parametrize(
    "preds,targets,mask,weights,task_weights,lt_mask,gt_mask,mse",
    [
        (
            torch.tensor([[-3, 2], [1, -1]], dtype=torch.float),
            torch.zeros([2, 2], dtype=torch.float),
            torch.ones([2, 2], dtype=torch.bool),
            torch.ones([2]),
            torch.ones([2]),
            torch.zeros([2, 2], dtype=torch.bool),
            torch.zeros([2, 2], dtype=torch.bool),
            torch.tensor(3.75000, dtype=torch.float),
        ),
        (
            torch.tensor([[-3, 2], [1, -1]], dtype=torch.float),
            torch.zeros([2, 2], dtype=torch.float),
            torch.ones([2, 2], dtype=torch.bool),
            torch.ones([2]),
            torch.ones([2]),
            torch.zeros([2, 2], dtype=torch.bool),
            torch.ones([2, 2], dtype=torch.bool),
            torch.tensor(2.5000, dtype=torch.float),
        ),
        (
            torch.tensor([[-3, 2], [1, -1]], dtype=torch.float),
            torch.zeros([2, 2], dtype=torch.float),
            torch.ones([2, 2], dtype=torch.bool),
            torch.ones([2]),
            torch.ones([2]),
            torch.ones([2, 2], dtype=torch.bool),
            torch.zeros([2, 2], dtype=torch.bool),
            torch.tensor(1.25000, dtype=torch.float),
        ),
    ],
)
def test_BoundedMSE(preds, targets, mask, weights, task_weights, lt_mask, gt_mask, mse):
    """
    Testing the bounded_mse loss function
    """
    bmse_loss = BoundedMSELoss(task_weights)
    loss = bmse_loss(preds, targets, mask, weights, lt_mask, gt_mask)
    torch.testing.assert_close(loss, mse)


@pytest.mark.parametrize(
    "preds,targets,mask,weights,task_weights,lt_mask,gt_mask,likelihood",
    [
        (
            torch.tensor([[0, 1]], dtype=torch.float),
            torch.zeros([1, 1]),
            torch.ones([1, 2], dtype=torch.bool),
            torch.ones([1]),
            torch.ones([2]),
            torch.zeros([2], dtype=torch.bool),
            torch.zeros([2], dtype=torch.bool),
            torch.tensor(0.39894228, dtype=torch.float),
        )
    ],
)
def test_MVE(preds, targets, mask, weights, task_weights, lt_mask, gt_mask, likelihood):
    """
    Tests the normal_mve loss function
    """
    mve_loss = MVELoss(task_weights)
    nll_calc = mve_loss(preds, targets, mask, weights, lt_mask, gt_mask)
    likelihood_calc = np.exp(-1 * nll_calc)
    torch.testing.assert_close(likelihood_calc, likelihood)


@pytest.mark.parametrize(
    "preds,targets,mask,weights,task_weights,lt_mask,gt_mask,v_kl,expected_loss",
    [
        (
            torch.tensor([[2, 2]]),
            torch.ones([1, 1]),
            torch.ones([1, 2], dtype=torch.bool),
            torch.ones([1]),
            torch.ones([1]),
            torch.zeros([1], dtype=torch.bool),
            torch.zeros([1], dtype=torch.bool),
            0,
            torch.tensor(0.6, dtype=torch.float),
        ),
        (
            torch.tensor([[2, 2]]),
            torch.ones([1, 1]),
            torch.ones([1, 2], dtype=torch.bool),
            torch.ones([1]),
            torch.ones([1]),
            torch.zeros([1], dtype=torch.bool),
            torch.zeros([1], dtype=torch.bool),
            0.2,
            torch.tensor(0.63862943, dtype=torch.float),
        ),
    ],
)
def test_BinaryDirichlet(
    preds, targets, mask, weights, task_weights, lt_mask, gt_mask, v_kl, expected_loss
):
    """
    Test on the dirichlet loss function for classification.
    Note these values were not hand derived, just testing for
    dimensional consistency.
    """
    binary_dirichlet_loss = BinaryDirichletLoss(task_weights=task_weights, v_kl=v_kl)
    loss = binary_dirichlet_loss(preds, targets, mask, weights, lt_mask, gt_mask)
    torch.testing.assert_close(loss, expected_loss)


@pytest.mark.parametrize(
    "preds,targets,mask,weights,task_weights,lt_mask,gt_mask,",
    [
        (
            torch.ones([1, 1]),
            torch.ones([1, 1]),
            torch.ones([1, 2], dtype=torch.bool),
            torch.ones([1]),
            torch.ones([1]),
            torch.zeros([1], dtype=torch.bool),
            torch.zeros([1], dtype=torch.bool),
        )
    ],
)
def test_BinaryDirichlet_wrong_dimensions(
    preds, targets, mask, weights, task_weights, lt_mask, gt_mask
):
    """
    Test on the dirichlet loss function for classification
    for dimension errors.
    """
    with pytest.raises(RuntimeError):
        binary_dirichlet_loss = BinaryDirichletLoss(task_weights)
        binary_dirichlet_loss(preds, targets, mask, weights, lt_mask, gt_mask)


@pytest.mark.parametrize(
    "preds,targets,mask,weights,task_weights,lt_mask,gt_mask,v_kl,expected_loss",
    [
        (
            torch.tensor([[[0.2, 0.1, 0.3], [0.1, 0.3, 0.1]], [[1.2, 0.5, 1.7], [1.1, 1.4, 0.8]]]),
            torch.tensor([[0, 0], [1, 1]]),
            torch.ones([2, 2], dtype=torch.bool),
            torch.ones([2]),
            torch.ones([2]),
            torch.zeros([2], dtype=torch.bool),
            torch.zeros([2], dtype=torch.bool),
            0.2,
            torch.tensor(1.868991, dtype=torch.float),
        ),
        (
            torch.tensor([[[0.2, 0.1, 0.3], [0.1, 0.3, 0.1]], [[1.2, 0.5, 1.7], [1.1, 1.4, 0.8]]]),
            torch.tensor([[0, 0], [1, 1]]),
            torch.ones([2, 2], dtype=torch.bool),
            torch.ones([2]),
            torch.ones([2]),
            torch.zeros([2], dtype=torch.bool),
            torch.zeros([2], dtype=torch.bool),
            0.0,
            torch.tensor(1.102344, dtype=torch.float),
        ),
    ],
)
def test_MulticlassDirichlet(
    preds, targets, mask, weights, task_weights, lt_mask, gt_mask, v_kl, expected_loss
):
    """
    Test on the dirichlet loss function for classification.
    Note these values were not hand derived, just testing for
    dimensional consistency.
    """
    multiclass_dirichlet_loss = MulticlassDirichletLoss(task_weights=task_weights, v_kl=v_kl)
    loss = multiclass_dirichlet_loss(preds, targets, mask, weights, lt_mask, gt_mask)
    torch.testing.assert_close(loss, expected_loss)


@pytest.mark.parametrize(
    "preds,targets,mask,weights,task_weights,lt_mask,gt_mask,v_kl,expected_loss",
    [
        (
            torch.tensor([[2, 2, 2, 2]]),
            torch.ones([1, 1]),
            torch.ones([1, 1], dtype=torch.bool),
            torch.ones([1]),
            torch.ones([1]),
            torch.zeros([1], dtype=torch.bool),
            torch.zeros([1], dtype=torch.bool),
            0,
            torch.tensor(1.56893861, dtype=torch.float),
        ),
        (
            torch.tensor([[2, 2, 2, 2]]),
            torch.ones([1, 1]),
            torch.ones([1, 1], dtype=torch.bool),
            torch.ones([1]),
            torch.ones([1]),
            torch.zeros([1], dtype=torch.bool),
            torch.zeros([1], dtype=torch.bool),
            0.2,
            torch.tensor(2.768938541, dtype=torch.float),
        ),
    ],
)
def test_Evidential(
    preds, targets, mask, weights, task_weights, lt_mask, gt_mask, v_kl, expected_loss
):
    """
    Test on the evidential loss function for classification.
    Note these values were not hand derived, just testing for
    dimensional consistency.
    """
    evidential_loss = EvidentialLoss(task_weights=task_weights, v_kl=v_kl)
    loss = evidential_loss(preds, targets, mask, weights, lt_mask, gt_mask)
    torch.testing.assert_close(loss, expected_loss)


@pytest.mark.parametrize(
    "preds,targets,mask,weights,task_weights,lt_mask,gt_mask",
    [
        (
            torch.ones([2, 2]),
            torch.ones([2, 2]),
            torch.ones([1, 1], dtype=torch.bool),
            torch.ones([1]),
            torch.ones([1]),
            torch.zeros([1], dtype=torch.bool),
            torch.zeros([1], dtype=torch.bool),
        )
    ],
)
def test_Evidential_wrong_dimensions(preds, targets, mask, weights, task_weights, lt_mask, gt_mask):
    """
    Test on the Evidential loss function for classification
    for dimension errors.
    """
    evidential_loss = EvidentialLoss(task_weights)
    with pytest.raises(ValueError):
        evidential_loss(preds, targets, mask, weights, lt_mask, gt_mask)


@pytest.mark.parametrize(
    "preds,targets,mask,weights,task_weights,lt_mask,gt_mask,expected_loss",
    [
        (
            torch.tensor([2, 2], dtype=torch.float),
            torch.ones([2], dtype=torch.float),
            torch.ones([2], dtype=torch.bool),
            torch.ones([1]),
            torch.ones([2]),
            torch.zeros([2], dtype=torch.bool),
            torch.zeros([2], dtype=torch.bool),
            torch.tensor(0.126928, dtype=torch.float),
        ),
        (
            torch.tensor([0.5, 0.5], dtype=torch.float),
            torch.ones([2], dtype=torch.float),
            torch.ones([2], dtype=torch.bool),
            torch.ones([1]),
            torch.ones([2]),
            torch.zeros([2], dtype=torch.bool),
            torch.zeros([2], dtype=torch.bool),
            torch.tensor(0.474077, dtype=torch.float),
        ),
    ],
)
def test_BCE(preds, targets, mask, weights, task_weights, lt_mask, gt_mask, expected_loss):
    """
    Test on the BCE loss function for classification.
    """
    bce_loss = BCELoss(task_weights)
    loss = bce_loss(preds, targets, mask, weights, lt_mask, gt_mask)
    torch.testing.assert_close(loss, expected_loss)


@pytest.mark.parametrize(
    "preds,targets,mask,weights,task_weights,lt_mask,gt_mask,expected_loss",
    [
        (
            torch.tensor([[[1.2, 0.5, 0.7], [-0.1, 0.3, 0.1]], [[1.2, 0.5, 0.7], [1.1, 1.3, 1.1]]]),
            torch.tensor([[1, 0], [1, 2]]),
            torch.ones([2, 2], dtype=torch.bool),
            torch.ones([2]),
            torch.ones([2]),
            torch.ones([2, 2], dtype=torch.bool),
            torch.ones([2, 2], dtype=torch.bool),
            torch.tensor(1.34214, dtype=torch.float),
        ),
        (
            torch.tensor([[[1.2, 1.5, 0.7], [-0.1, 2.3, 1.1]], [[1.2, 1.5, 1.7], [2.1, 1.3, 1.1]]]),
            torch.tensor([[1, 1], [2, 2]], dtype=torch.float64),
            torch.ones([2, 2], dtype=torch.bool),
            torch.ones([2]),
            torch.ones([2]),
            torch.ones([2, 2], dtype=torch.bool),
            torch.ones([2, 2], dtype=torch.bool),
            torch.tensor(0.899472, dtype=torch.float),
        ),
    ],
)
def test_CrossEntropy(preds, targets, mask, weights, task_weights, lt_mask, gt_mask, expected_loss):
    """
    Test on the CE loss function for classification.
    Note these values were not hand derived, just testing for
    dimensional consistency.
    """
    cross_entropy_loss = CrossEntropyLoss(task_weights)
    loss = cross_entropy_loss(preds, targets, mask, weights, lt_mask, gt_mask)
    torch.testing.assert_close(loss, expected_loss)


@pytest.mark.parametrize(
    "preds,targets,mask,weights,task_weights,lt_mask,gt_mask,expected_loss",
    [
        (
            torch.tensor([0, 1, 1, 0]),
            torch.tensor([0, 1, 1, 0]),
            torch.ones([4], dtype=torch.bool),
            torch.ones(1),
            torch.ones(4),
            torch.zeros([1, 4], dtype=torch.bool),
            torch.zeros([1, 4], dtype=torch.bool),
            torch.tensor(0, dtype=torch.float),
        ),
        (
            torch.tensor([0, 1, 0, 1, 1, 1, 0, 1, 1]),
            torch.tensor([0, 1, 1, 0, 1, 1, 0, 0, 1]),
            torch.ones([9], dtype=torch.bool),
            torch.ones(1),
            torch.ones(9),
            torch.zeros([1, 9], dtype=torch.bool),
            torch.zeros([1, 9], dtype=torch.bool),
            torch.tensor(0.683772, dtype=torch.float),
        ),
    ],
)
def test_BinaryMCC(preds, targets, mask, weights, task_weights, lt_mask, gt_mask, expected_loss):
    """
    Test on the BinaryMCC loss function for classification. Values have been checked using TorchMetrics.
    """
    binary_mcc_loss = BinaryMCCLoss(task_weights)
    loss = binary_mcc_loss(preds, targets, mask, weights, lt_mask, gt_mask)
    torch.testing.assert_close(loss, expected_loss)


@pytest.mark.parametrize(
    "preds,targets,mask,weights,task_weights,lt_mask,gt_mask,expected_loss",
    [
        (
            torch.tensor(
                [[[0.16, 0.26, 0.58], [0.22, 0.61, 0.17]], [[0.71, 0.09, 0.20], [0.05, 0.82, 0.13]]]
            ),
            torch.tensor([[2, 1], [0, 0]]),
            torch.ones([2, 2], dtype=torch.bool),
            torch.ones([1, 2]),
            torch.ones([2]),
            torch.zeros([2, 2], dtype=torch.bool),
            torch.zeros([2, 2], dtype=torch.bool),
            torch.tensor(0.2697033, dtype=torch.float),
        ),
        (
            torch.tensor(
                [[[0.16, 0.26, 0.58], [0.22, 0.61, 0.17]], [[0.71, 0.09, 0.20], [0.05, 0.82, 0.13]]]
            ),
            torch.tensor([[2, 1], [0, 0]]),
            torch.tensor([[1, 1], [0, 1]], dtype=torch.bool),
            torch.ones([1, 2]),
            torch.ones([2]),
            torch.zeros([2, 2], dtype=bool),
            torch.zeros([2, 2], dtype=bool),
            torch.tensor(0.3876276, dtype=torch.float),
        ),
    ],
)
def test_MulticlassMCC(
    preds, targets, mask, weights, task_weights, lt_mask, gt_mask, expected_loss
):
    """
    Test on the MulticlassMCC loss function for classification.
    """
    multiclass_mcc_loss = MulticlassMCCLoss(task_weights)
    loss = multiclass_mcc_loss(preds, targets, mask, weights, lt_mask, gt_mask)
    torch.testing.assert_close(loss, expected_loss)


@pytest.mark.parametrize(
    "preds,targets,mask,weights,task_weights,lt_mask,gt_mask,threshold,expected_loss",
    [
        (
            torch.tensor([[0.8, 0.2], [0.3, 0.7]]),
            torch.tensor([[0.9, 0.1], [0.4, 0.6]]),
            torch.ones([2, 2], dtype=torch.bool),
            torch.ones([1]),
            torch.ones([2]),
            torch.ones([2], dtype=torch.bool),
            torch.ones([2], dtype=torch.bool),
            None,
            torch.tensor(0.031319, dtype=torch.float),
        ),
        (
            torch.tensor([[0.6, 0.4], [0.2, 0.8]]),
            torch.tensor([[0.7, 0.3], [0.3, 0.7]]),
            torch.tensor([[1, 1], [1, 0]], dtype=torch.bool),
            torch.ones([1]),
            torch.ones([2]),
            torch.ones([2], dtype=torch.bool),
            torch.ones([2], dtype=torch.bool),
            None,
            torch.tensor(0.295655, dtype=torch.float),
        ),
        (
            torch.tensor([[0.6, 0.4], [0.2, 0.8]]),
            torch.tensor([[0.7, 0.3], [0.3, 0.7]]),
            torch.tensor([[1, 1], [1, 1]], dtype=torch.bool),
            torch.ones([1]),
            torch.ones([2]),
            torch.ones([2], dtype=torch.bool),
            torch.ones([2], dtype=torch.bool),
            0.5,
            torch.tensor(0.033673, dtype=torch.float),
        ),
    ],
)
def test_SID(
    preds, targets, mask, weights, task_weights, lt_mask, gt_mask, threshold, expected_loss
):
    """
    Test on the SID loss function. These values were not handchecked,
    just checking function returns values with/without mask and threshold.
    """
    sid_loss = SIDLoss(task_weights=task_weights, threshold=threshold)
    loss = sid_loss(preds, targets, mask, weights, lt_mask, gt_mask)
    torch.testing.assert_close(loss, expected_loss)


@pytest.mark.parametrize(
    "preds,targets,mask,weights,task_weights,lt_mask,gt_mask,threshold,expected_loss",
    [
        (
            torch.tensor([[0.1, 0.3, 0.5, 0.7], [0.2, 0.4, 0.6, 0.8]]),
            torch.tensor([[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]]),
            torch.tensor([[1, 1, 1, 1], [1, 0, 1, 0]], dtype=torch.bool),
            torch.ones([2, 1]),
            torch.ones([1, 4]),
            torch.zeros([2, 4], dtype=torch.bool),
            torch.zeros([2, 4], dtype=torch.bool),
            None,
            torch.tensor(0.1125, dtype=torch.float),
        ),
        (
            torch.tensor([[0.1, 0.3, 0.5, 0.7], [0.2, 0.4, 0.6, 0.8]]),
            torch.tensor([[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]]),
            torch.ones([2, 4], dtype=torch.bool),
            torch.ones([2, 1]),
            torch.ones([1, 4]),
            torch.zeros([2, 4], dtype=torch.bool),
            torch.zeros([2, 4], dtype=torch.bool),
            None,
            torch.tensor(0.515625, dtype=torch.float),
        ),
        (
            torch.tensor([[0.1, 0.3, 0.5, 0.7], [0.2, 0.4, 0.6, 0.8]]),
            torch.tensor([[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]]),
            torch.ones([2, 4], dtype=torch.bool),
            torch.ones([2, 1]),
            torch.ones([1, 4]),
            torch.zeros([2, 4], dtype=torch.bool),
            torch.zeros([2, 4], dtype=torch.bool),
            0.3,
            torch.tensor(0.501984, dtype=torch.float),
        ),
    ],
)
def test_Wasserstein(
    preds, targets, mask, weights, task_weights, lt_mask, gt_mask, threshold, expected_loss
):
    """
    Test on the Wasserstein loss function. These values were not handchecked,
    just checking function returns values with/without mask and threshold.
    """
    wasserstein_loss = WassersteinLoss(task_weights=task_weights, threshold=threshold)
    loss = wasserstein_loss(preds, targets, mask, weights, lt_mask, gt_mask)
    torch.testing.assert_close(loss, expected_loss)
