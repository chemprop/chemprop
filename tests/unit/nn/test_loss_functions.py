"""Chemprop unit tests for chemprop/models/loss.py"""

import numpy as np
import pytest
import torch

from chemprop.nn.metrics import (
    SID,
    BCELoss,
    BinaryMCCLoss,
    BoundedMSE,
    CrossEntropyLoss,
    DirichletLoss,
    EvidentialLoss,
    MulticlassMCCLoss,
    MVELoss,
    PointQuantileLoss,
    Wasserstein,
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
    bmse_loss = BoundedMSE(task_weights)
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
            torch.tensor([[[2, 2]]]),
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
            torch.tensor([[[2, 2]]]),
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
    binary_dirichlet_loss = DirichletLoss(task_weights=task_weights, v_kl=v_kl)
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
    with pytest.raises(IndexError):
        binary_dirichlet_loss = DirichletLoss(task_weights)
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
    multiclass_dirichlet_loss = DirichletLoss(task_weights=task_weights, v_kl=v_kl)
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
            torch.ones([2]),
            torch.ones([2]),
            torch.zeros([2, 2], dtype=torch.bool),
            torch.zeros([2, 2], dtype=torch.bool),
            torch.tensor(0.5, dtype=torch.float),
        ),
        (
            torch.tensor(
                [[[0.16, 0.26, 0.58], [0.22, 0.61, 0.17]], [[0.71, 0.09, 0.20], [0.05, 0.82, 0.13]]]
            ),
            torch.tensor([[2, 1], [0, 0]]),
            torch.tensor([[1, 1], [0, 1]], dtype=torch.bool),
            torch.ones([2]),
            torch.ones([2]),
            torch.zeros([2, 2], dtype=bool),
            torch.zeros([2, 2], dtype=bool),
            torch.tensor(1.0, dtype=torch.float),
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
    sid_loss = SID(task_weights=task_weights, threshold=threshold)
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
    wasserstein_loss = Wasserstein(task_weights=task_weights, threshold=threshold)
    loss = wasserstein_loss(preds, targets, mask, weights, lt_mask, gt_mask)
    torch.testing.assert_close(loss, expected_loss)


@pytest.mark.parametrize(
    "preds,targets,mask,weights,task_weights,lt_mask,gt_mask,alpha,expected_loss",
    [
        # Basic test: prediction below target (diff > 0)
        (
            torch.tensor([[1.0], [2.0]], dtype=torch.float),
            torch.tensor([[2.0], [3.0]], dtype=torch.float),
            torch.ones([2, 1], dtype=torch.bool),
            torch.ones([2]),
            torch.ones([1]),
            torch.zeros([2, 1], dtype=torch.bool),
            torch.zeros([2, 1], dtype=torch.bool),
            0.1,
            torch.tensor(0.1, dtype=torch.float),  # alpha * (2-1) = 0.1
        ),
        # Basic test: prediction above target (diff < 0)
        (
            torch.tensor([[2.0], [3.0]], dtype=torch.float),
            torch.tensor([[1.0], [2.0]], dtype=torch.float),
            torch.ones([2, 1], dtype=torch.bool),
            torch.ones([2]),
            torch.ones([1]),
            torch.zeros([2, 1], dtype=torch.bool),
            torch.zeros([2, 1], dtype=torch.bool),
            0.1,
            torch.tensor(0.9, dtype=torch.float),  # (1-alpha) * (2-1) = 0.9
        ),
        # Perfect prediction (diff = 0)
        (
            torch.tensor([[1.0], [2.0]], dtype=torch.float),
            torch.tensor([[1.0], [2.0]], dtype=torch.float),
            torch.ones([2, 1], dtype=torch.bool),
            torch.ones([2]),
            torch.ones([1]),
            torch.zeros([2, 1], dtype=torch.bool),
            torch.zeros([2, 1], dtype=torch.bool),
            0.1,
            torch.tensor(0.0, dtype=torch.float),
        ),
        # Mixed case: one above, one below
        (
            torch.tensor([[1.0], [3.0]], dtype=torch.float),
            torch.tensor([[2.0], [2.0]], dtype=torch.float),
            torch.ones([2, 1], dtype=torch.bool),
            torch.ones([2]),
            torch.ones([1]),
            torch.zeros([2, 1], dtype=torch.bool),
            torch.zeros([2, 1], dtype=torch.bool),
            0.1,
            torch.tensor(0.5, dtype=torch.float),  # (0.1*1.0 + 0.9*1.0)/2 = 0.5
        ),
        # Different alpha value (median regression, alpha=0.5)
        (
            torch.tensor([[1.0], [3.0]], dtype=torch.float),
            torch.tensor([[2.0], [2.0]], dtype=torch.float),
            torch.ones([2, 1], dtype=torch.bool),
            torch.ones([2]),
            torch.ones([1]),
            torch.zeros([2, 1], dtype=torch.bool),
            torch.zeros([2, 1], dtype=torch.bool),
            0.5,
            torch.tensor(0.5, dtype=torch.float),  # (0.5*1 + 0.5*1)/2 = 0.5
        ),
        # Multiple tasks
        (
            torch.tensor([[1.0, 2.0], [3.0, 1.0]], dtype=torch.float),
            torch.tensor([[2.0, 1.0], [2.0, 2.0]], dtype=torch.float),
            torch.ones([2, 2], dtype=torch.bool),
            torch.ones([2]),
            torch.ones([2]),
            torch.zeros([2, 2], dtype=torch.bool),
            torch.zeros([2, 2], dtype=torch.bool),
            0.1,
            torch.tensor(0.5, dtype=torch.float),  # Average across all tasks
        ),
    ],
)
def test_PointQuantileLoss(
    preds, targets, mask, weights, task_weights, lt_mask, gt_mask, alpha, expected_loss
):
    """
    Test the PointQuantileLoss (pinball loss) function.
    The pinball loss is: alpha * max(0, target - pred) + (1-alpha) * max(0, pred - target)
    """
    quantile_loss = PointQuantileLoss(task_weights=task_weights, alpha=alpha)
    loss = quantile_loss(preds, targets, mask, weights, lt_mask, gt_mask)
    torch.testing.assert_close(loss, expected_loss, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize(
    "preds,targets,mask,weights,task_weights,lt_mask,gt_mask,alpha",
    [
        # Test with mask
        (
            torch.tensor([[1.0], [2.0], [3.0]], dtype=torch.float),
            torch.tensor([[2.0], [3.0], [4.0]], dtype=torch.float),
            torch.tensor([[1], [1], [0]], dtype=torch.bool),
            torch.ones([3]),
            torch.ones([1]),
            torch.zeros([3, 1], dtype=torch.bool),
            torch.zeros([3, 1], dtype=torch.bool),
            0.1,
        ),
        # Test with sample weights
        (
            torch.tensor([[1.0], [2.0]], dtype=torch.float),
            torch.tensor([[2.0], [3.0]], dtype=torch.float),
            torch.ones([2, 1], dtype=torch.bool),
            torch.tensor([0.5, 2.0]),
            torch.ones([1]),
            torch.zeros([2, 1], dtype=torch.bool),
            torch.zeros([2, 1], dtype=torch.bool),
            0.1,
        ),
        # Test with task weights
        (
            torch.tensor([[1.0, 2.0]], dtype=torch.float),
            torch.tensor([[2.0, 3.0]], dtype=torch.float),
            torch.ones([1, 2], dtype=torch.bool),
            torch.ones([1]),
            torch.tensor([0.5, 2.0]),
            torch.zeros([1, 2], dtype=torch.bool),
            torch.zeros([1, 2], dtype=torch.bool),
            0.1,
        ),
    ],
)
def test_PointQuantileLoss_with_masks_and_weights(
    preds, targets, mask, weights, task_weights, lt_mask, gt_mask, alpha
):
    """
    Test PointQuantileLoss with masks and weights.
    The loss should be properly weighted and masked.
    """
    quantile_loss = PointQuantileLoss(task_weights=task_weights, alpha=alpha)
    loss = quantile_loss(preds, targets, mask, weights, lt_mask, gt_mask)

    # Loss should be a scalar (averaged)
    assert loss.numel() == 1
    assert loss.item() >= 0  # Loss should be non-negative


@pytest.mark.parametrize("alpha", [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0])
def test_PointQuantileLoss_alpha_values(alpha):
    """
    Test PointQuantileLoss with different alpha values.
    Alpha should be in [0, 1] and represents the quantile level.
    """
    preds = torch.tensor([[1.0], [3.0]], dtype=torch.float)
    targets = torch.tensor([[2.0], [2.0]], dtype=torch.float)
    mask = torch.ones([2, 1], dtype=torch.bool)
    weights = torch.ones([2])
    task_weights = torch.ones([1])
    lt_mask = torch.zeros([2, 1], dtype=torch.bool)
    gt_mask = torch.zeros([2, 1], dtype=torch.bool)

    quantile_loss = PointQuantileLoss(task_weights=task_weights, alpha=alpha)
    loss = quantile_loss(preds, targets, mask, weights, lt_mask, gt_mask)

    # Loss should be non-negative
    assert loss.item() >= 0

    # For alpha=0.5 (median), loss should be symmetric
    if alpha == 0.5:
        # Swapping preds should give same loss
        preds_swapped = torch.tensor([[3.0], [1.0]], dtype=torch.float)
        loss_swapped = quantile_loss(preds_swapped, targets, mask, weights, lt_mask, gt_mask)
        torch.testing.assert_close(loss, loss_swapped, rtol=1e-5, atol=1e-5)


def test_PointQuantileLoss_extra_repr():
    """
    Test that extra_repr returns the correct string representation.
    """
    quantile_loss = PointQuantileLoss(alpha=0.25)
    repr_str = quantile_loss.extra_repr()
    assert "alpha=0.25" in repr_str


def test_PointQuantileLoss_mathematical_correctness():
    """
    Test mathematical correctness of the pinball loss formula.
    For a given alpha and difference d = target - pred:
    - If d > 0: loss = alpha * d
    - If d <= 0: loss = (1 - alpha) * (-d)
    """
    alpha = 0.2
    quantile_loss = PointQuantileLoss(alpha=alpha)

    # Case 1: pred < target (d > 0)
    preds = torch.tensor([[1.0]], dtype=torch.float)
    targets = torch.tensor([[3.0]], dtype=torch.float)
    mask = torch.ones([1, 1], dtype=torch.bool)
    weights = torch.ones([1])
    lt_mask = torch.zeros([1, 1], dtype=torch.bool)
    gt_mask = torch.zeros([1, 1], dtype=torch.bool)

    loss = quantile_loss(preds, targets, mask, weights, lt_mask, gt_mask)
    expected = alpha * (targets - preds).item()  # alpha * 2.0 = 0.4
    torch.testing.assert_close(loss, torch.tensor(expected), rtol=1e-5, atol=1e-5)

    # Case 2: pred > target (d < 0)
    preds = torch.tensor([[3.0]], dtype=torch.float)
    targets = torch.tensor([[1.0]], dtype=torch.float)

    loss = quantile_loss(preds, targets, mask, weights, lt_mask, gt_mask)
    expected = (1 - alpha) * (preds - targets).item()  # 0.8 * 2.0 = 1.6
    torch.testing.assert_close(loss, torch.tensor(expected), rtol=1e-5, atol=1e-5)

    # Case 3: pred == target (d = 0)
    preds = torch.tensor([[2.0]], dtype=torch.float)
    targets = torch.tensor([[2.0]], dtype=torch.float)

    loss = quantile_loss(preds, targets, mask, weights, lt_mask, gt_mask)
    torch.testing.assert_close(loss, torch.tensor(0.0), rtol=1e-5, atol=1e-5)
