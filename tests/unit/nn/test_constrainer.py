import pytest
import torch

from chemprop.nn import ConstrainerFFN


@pytest.mark.parametrize("fp_dim", [2, 100, 300, 600])
def test_constrainer_forward(fp_dim):
    batch = torch.tensor([0, 1, 1, 3, 3, 3])
    rows_per_group = torch.bincount(batch)
    b = len(batch)
    t = 3
    m = batch.max().item() + 1

    fp = torch.randn(b, fp_dim)
    preds = torch.randn(b, t)
    constraints = torch.randn(m, t)
    constraints[2] = 0

    constrainer = ConstrainerFFN(
        n_constraints=t, fp_dim=fp_dim, hidden_dim=32, n_layers=2, dropout=0.1, activation="relu"
    )
    with torch.no_grad():
        constrained_preds = constrainer(fp, preds, batch, constraints)
    constrained_preds = torch.split(constrained_preds, rows_per_group.tolist(), dim=0)
    constrained_preds = torch.stack([torch.sum(pred, dim=0) for pred in constrained_preds])

    assert torch.allclose(constrained_preds, constraints, rtol=1e-5, atol=1e-5)
