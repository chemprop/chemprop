import logging
from types import SimpleNamespace

import numpy as np
import pytest

from chemprop.cli.train import evaluate_and_save_predictions
from chemprop.nn.metrics import BoundedMAE, BoundedMSE


class _Dataset(SimpleNamespace):
    def __len__(self):
        return len(self.Y)


def _namespace_for_two_tasks():
    # targets are the bounds for rows flagged by lt/gt, so a "bounded" prediction on the
    # right side of the bound has zero error
    preds = np.array([[1.0, 5.0], [2.0, 5.0], [3.0, 5.0], [4.0, 5.0]])
    targets = np.array([[2.0, 4.0], [2.0, 4.0], [2.0, 4.0], [2.0, 4.0]])
    lt_mask = np.array([[True, False], [False, False], [False, False], [False, False]])
    gt_mask = np.array([[False, True], [False, False], [False, False], [False, False]])
    dset = _Dataset(
        Y=targets,
        lt_mask=lt_mask,
        gt_mask=gt_mask,
        data=[SimpleNamespace(lt_mask=lt_mask[0], gt_mask=gt_mask[0])],
        names=["a", "b", "c", "d"],
    )
    args = SimpleNamespace(
        target_columns=["t0", "t1"],
        input_columns=["smiles"],
        task_type="regression",
        show_individual_scores=True,
    )
    return preds, SimpleNamespace(dataset=dset), args


@pytest.mark.parametrize(
    ("metric_cls", "task_errors"),
    [
        # task 0: errors 0, 0, 1, 2 (row 0 is bounded by "< 2")
        # task 1: errors 0, 1, 1, 1 (row 0 is bounded by "> 4")
        (BoundedMSE, ([0, 0, 1, 4], [0, 1, 1, 1])),
        (BoundedMAE, ([0, 0, 1, 2], [0, 1, 1, 1])),
    ],
)
def test_bounded_metrics_use_per_task_masks(tmp_path, caplog, metric_cls, task_errors):
    preds, loader, args = _namespace_for_two_tasks()
    expected = [np.mean(e) for e in task_errors]

    with caplog.at_level(logging.INFO):
        evaluate_and_save_predictions(preds, loader, [metric_cls()], tmp_path, args)

    alias = metric_cls.alias
    for col, exp in zip(args.target_columns, expected):
        line = next(m for m in caplog.messages if m.startswith(f"test/{col}/{alias}:"))
        assert float(line.split(":")[1].replace("tensor(", "").rstrip(")")) == pytest.approx(
            exp
        )
    mean_line = next(m for m in caplog.messages if m.startswith(f"test/{alias}:"))
    assert float(mean_line.split(":")[1].replace("tensor(", "").rstrip(")")) == pytest.approx(
        np.mean(expected)
    )
