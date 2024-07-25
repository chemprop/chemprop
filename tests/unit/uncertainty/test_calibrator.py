# import pytest
# import torch

# from chemprop.uncertainty.calibrator import (
#     ConformalAdaptiveMulticlassCalibrator,
#     ConformalMulticlassCalibrator,
#     ConformalMultilabelCalibrator,
#     ConformalQuantileRegressionCalibrator,
#     ConformalRegressionCalibrator,
#     IsotonicCalibrator,
#     IsotonicMulticlassCalibrator,
#     MVEWeightingCalibrator,
#     PlattCalibrator,
#     TScalingCalibrator,
#     ZelikmanCalibrator,
#     ZScalingCalibrator,
# )

# # Example
# @pytest.mark.parametrize(
#     "cal_preds,cal_uncs,cal_targets,cal_mask,test_preds,test_uncs,cal_test_preds,cal_test_uncs",
#     [
#         (
#             torch.tensor([[-3, 2], [1, -1]], dtype=torch.float),
#             torch.tensor([[0.3, 0.2], [0.1, 0.1]], dtype=torch.float),
#             torch.zeros([2, 2], dtype=torch.int),
#             torch.ones([2, 2], dtype=torch.bool),
#             torch.tensor([[4, 2], [5, -2]], dtype=torch.float),
#             torch.tensor([[0.1, 0.2], [0.3, 0.4]], dtype=torch.float),
#             torch.tensor([[4, 2], [5, -2]], dtype=torch.float),
#             torch.tensor([[5.8, 4], [6, 4.2]], dtype=torch.float),
#         ),
#         ...,
#     ],
# )
# def test_ConformalQuantileRegressionCalibrator(
#     cal_preds, cal_uncs, cal_targets, cal_mask, test_preds, test_uncs, cal_test_preds, cal_test_uncs
# ):
#     """
#     Testing the ConformalQuantileRegressionCalibrator
#     """
#     calibrator = ConformalQuantileRegressionCalibrator(conformal_alpha=0.1)
#     calibrator.calibrate(cal_preds, cal_uncs, cal_targets, cal_mask)
#     preds, uncs = calibrator.apply_calibration(test_preds, test_uncs)

#     torch.testing.assert_close(preds, cal_test_preds)
#     torch.testing.assert_close(uncs, cal_test_uncs)
