import numpy as np
import pytest
from sklearn.metrics import matthews_corrcoef
import torch

from chemprop.nn.metrics import BinaryMCCLoss, BinaryMCCMetric


@pytest.mark.parametrize("metric_cls", [BinaryMCCLoss, BinaryMCCMetric])
@pytest.mark.parametrize("n_tasks", [1, 2, 5])
@pytest.mark.parametrize("weight_shape", ["default", "vector", "column"])
def test_binary_mcc_sample_weights(metric_cls, n_tasks, weight_shape):
    # Cover both unequal batch/task sizes and the silent broadcasting case b == t.
    targets = torch.tensor([0, 1, 1, 0, 1]).unsqueeze(1).repeat(1, n_tasks)
    preds = torch.tensor([0, 1, 0, 1, 1], dtype=torch.float).unsqueeze(1).repeat(1, n_tasks)
    mask = torch.ones_like(targets, dtype=torch.bool)
    for task in range(n_tasks):
        mask[task, task] = False
    sample_weights = torch.tensor([1.0, 2.0, 4.0, 8.0, 16.0])
    task_weights = torch.arange(1, n_tasks + 1, dtype=torch.float)
    if weight_shape == "default":
        sample_weights = torch.ones(5)
        weights = None
    else:
        weights = sample_weights if weight_shape == "vector" else sample_weights.unsqueeze(1)

    expected = np.mean(
        [
            matthews_corrcoef(
                targets[mask[:, task], task],
                preds[mask[:, task], task],
                sample_weight=sample_weights[mask[:, task]],
            )
            * task_weights[task].item()
            for task in range(n_tasks)
        ]
    )
    if metric_cls is BinaryMCCLoss:
        expected = 1 - expected

    metric = metric_cls(task_weights)
    actual = metric(preds, targets, mask, weights)
    assert actual.item() == pytest.approx(expected, abs=1e-6)

    # Epoch accumulation must give the same result with uneven batch sizes.
    metric.reset()
    for start, stop in [(0, 2), (2, 5)]:
        batch_weights = None if weights is None else weights[start:stop]
        metric.update(preds[start:stop], targets[start:stop], mask[start:stop], batch_weights)
    assert metric.compute().item() == pytest.approx(expected, abs=1e-6)


@pytest.mark.parametrize("metric_cls", [BinaryMCCLoss, BinaryMCCMetric])
@pytest.mark.parametrize("weight_shape", ["vector", "column"])
@pytest.mark.parametrize("use_logits", [False, True])
def test_binary_mcc_soft_weights_match_repeated_samples(metric_cls, weight_shape, use_logits):
    values = torch.tensor([[-2.0, 1.0], [0.5, -1.0], [2.0, -0.5]])
    preds = (values if use_logits else values.sigmoid()).requires_grad_()
    targets = torch.tensor([[0, 1], [1, 0], [0, 1]])
    mask = torch.tensor([[True, False], [True, True], [True, True]])
    repeats = torch.tensor([1, 2, 3])
    weights = repeats.float()
    if weight_shape == "column":
        weights = weights.unsqueeze(1)
    task_weights = [0.5, 1.5]

    actual = metric_cls(task_weights)(preds, targets, mask, weights)
    # Integer sample weights are equivalent to repeating observations. Explicit column
    # weights on the reference avoid the default-weight broadcasting path under test.
    expected = metric_cls(task_weights)(
        preds.repeat_interleave(repeats, dim=0),
        targets.repeat_interleave(repeats, dim=0),
        mask.repeat_interleave(repeats, dim=0),
        torch.ones((repeats.sum(), 1)),
    )
    torch.testing.assert_close(actual, expected)
    actual_grad = torch.autograd.grad(actual, preds, retain_graph=True)[0]
    expected_grad = torch.autograd.grad(expected, preds)[0]
    torch.testing.assert_close(actual_grad, expected_grad)
    assert torch.isfinite(actual_grad).all()
