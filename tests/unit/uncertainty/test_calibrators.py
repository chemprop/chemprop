import pytest
import torch

# TODO: adding IsotonicCalibrator, IsotonicMulticlassCalibrator, MVEWeightingCalibrator, PlattCalibrator, TScalingCalibrator, ZelikmanCalibrator, ZScalingCalibrator,
from chemprop.uncertainty.calibrator import (
    ConformalAdaptiveMulticlassCalibrator,
    ConformalMulticlassCalibrator,
    ConformalMultilabelCalibrator,
    ConformalRegressionCalibrator,
)


@pytest.mark.parametrize(
    "cal_preds,cal_uncs,cal_targets,cal_mask,test_preds,test_uncs,cal_test_uncs",
    [
        (
            None,
            torch.tensor(
                [
                    [[0.2, 0.3, 0.5], [0.1, 0.6, 0.3]],
                    [[0.1, 0.6, 0.3], [0.4, 0.4, 0.2]],
                    [[0.4, 0.4, 0.2], [0.2, 0.3, 0.5]],
                ]
            ),
            torch.tensor([[2, 1], [1, 0], [0, 2]]).long(),
            torch.ones([3, 2], dtype=torch.bool),
            None,
            torch.tensor(
                [
                    [[0.3, 0.4, 0.3], [0.5, 0.2, 0.3]],
                    [[0.5, 0.2, 0.3], [0.6, 0.3, 0.1]],
                    [[0.6, 0.3, 0.1], [0.3, 0.4, 0.3]],
                ]
            ),
            torch.tensor(
                [[[0, 1, 0], [1, 0, 0]], [[1, 0, 0], [1, 0, 0]], [[1, 0, 0], [0, 1, 0]]]
            ).int(),
        )
    ],
)
def test_ConformalAdaptiveMulticlassCalibrator(
    cal_preds, cal_uncs, cal_targets, cal_mask, test_preds, test_uncs, cal_test_uncs
):
    """
    Testing the ConformalAdaptiveMulticlassCalibrator
    """
    calibrator = ConformalAdaptiveMulticlassCalibrator(alpha=0.5)
    calibrator.fit(cal_preds, cal_uncs, cal_targets, cal_mask)
    preds, uncs = calibrator.apply(test_preds, test_uncs)

    torch.testing.assert_close(preds, test_preds)
    torch.testing.assert_close(uncs, cal_test_uncs)


@pytest.mark.parametrize(
    "cal_preds,cal_uncs,cal_targets,cal_mask,test_preds,test_uncs,cal_test_uncs",
    [
        (
            None,
            torch.tensor(
                [
                    [[0.2, 0.3, 0.5], [0.1, 0.6, 0.3]],
                    [[0.1, 0.6, 0.3], [0.4, 0.4, 0.2]],
                    [[0.4, 0.4, 0.2], [0.2, 0.3, 0.5]],
                ]
            ),
            torch.tensor([[2, 2], [1, 0], [0, 2]]).long(),
            torch.ones([3, 2], dtype=torch.bool),
            None,
            torch.tensor(
                [
                    [[0.3, 0.4, 0.3], [0.5, 0.2, 0.3]],
                    [[0.5, 0.2, 0.3], [0.6, 0.3, 0.1]],
                    [[0.6, 0.3, 0.1], [0.3, 0.4, 0.3]],
                ]
            ),
            torch.tensor(
                [[[0, 1, 0], [1, 0, 1]], [[1, 0, 0], [1, 1, 0]], [[1, 0, 0], [1, 1, 1]]]
            ).int(),
        )
    ],
)
def test_ConformalMulticlassCalibrator(
    cal_preds, cal_uncs, cal_targets, cal_mask, test_preds, test_uncs, cal_test_uncs
):
    """
    Testing the ConformalMulticlassCalibrator
    """
    calibrator = ConformalMulticlassCalibrator(alpha=0.5)
    calibrator.fit(cal_preds, cal_uncs, cal_targets, cal_mask)
    preds, uncs = calibrator.apply(test_preds, test_uncs)

    torch.testing.assert_close(preds, test_preds)
    torch.testing.assert_close(uncs, cal_test_uncs)


@pytest.mark.parametrize(
    "cal_preds,cal_uncs,cal_targets,cal_mask,test_preds,test_uncs,cal_test_uncs",
    [
        (
            None,
            torch.tensor([[0, 1, 0], [1, 0, 0], [0, 0, 1]]),
            torch.tensor([[0, 1, 0], [1, 0, 0], [0, 0, 1]]),
            torch.ones([3, 3], dtype=torch.bool),
            None,
            torch.tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
            torch.tensor(
                [[1, 1, 1, 1, 0, 0], [1, 1, 1, 0, 1, 0], [1, 1, 1, 0, 0, 1]], dtype=torch.int
            ),
        )
    ],
)
def test_ConformalMultilabelCalibrator(
    cal_preds, cal_uncs, cal_targets, cal_mask, test_preds, test_uncs, cal_test_uncs
):
    """
    Testing the ConformalMultilabelCalibrator
    """
    calibrator = ConformalMultilabelCalibrator(alpha=0.1)
    calibrator.fit(cal_preds, cal_uncs, cal_targets, cal_mask)
    preds, uncs = calibrator.apply(test_preds, test_uncs)

    torch.testing.assert_close(preds, test_preds)
    torch.testing.assert_close(uncs, cal_test_uncs)


@pytest.mark.parametrize(
    "cal_preds,cal_uncs,cal_targets,cal_mask,test_preds,test_uncs,cal_test_uncs",
    [
        (
            torch.arange(100).unsqueeze(1),
            torch.arange(100).unsqueeze(1) / 10,
            torch.arange(10, 110).unsqueeze(1),
            torch.ones([100, 1], dtype=torch.bool),
            torch.arange(100, 200).unsqueeze(1),
            torch.arange(100, 200).unsqueeze(1) / 10,
            torch.arange(29.2, 39.1, 0.1).unsqueeze(1),
        ),
        (
            torch.arange(100).unsqueeze(1),
            torch.zeros(100, 1),
            torch.arange(10, 110).unsqueeze(1),
            torch.ones([100, 1], dtype=torch.bool),
            torch.arange(100, 200).unsqueeze(1),
            torch.zeros(100, 1),
            torch.ones(100, 1) * 20,
        ),
    ],
)
def test_ConformalRegressionCalibrator(
    cal_preds, cal_uncs, cal_targets, cal_mask, test_preds, test_uncs, cal_test_uncs
):
    """
    Testing the ConformalRegressionCalibrator
    """
    calibrator = ConformalRegressionCalibrator(alpha=0.1)
    calibrator.fit(cal_preds, cal_uncs, cal_targets, cal_mask)
    preds, uncs = calibrator.apply(test_preds, test_uncs)

    torch.testing.assert_close(preds, test_preds)
    torch.testing.assert_close(uncs, cal_test_uncs)
