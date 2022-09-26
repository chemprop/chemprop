import os

#alpha_list = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
alpha_list = [0.1]

for alpha in alpha_list:
    os.system(f"python3 ../predict.py --test_path data/delaney3class3.csv --checkpoint_dir quantile_regression/delaney_checkpoints_3class1_quantile_{alpha} --preds_path quantile_regression_new/delaney_preds_conformal_quantile_{alpha}.csv --calibration_method conformal_quantile_regression --calibration_path data/delaney3class2.csv --conformal_alpha {alpha}")

# Trains on 3class1, calibrates on 3class2, predicts 3class3

# python3 ../predict.py --test_path data/delaney3class3.csv --checkpoint_dir quantile_regression/delaney_checkpoints_3class1_quantile_0.1 --preds_path quantile_regression_new/delaney_preds_conformal_quantile_0.1_nocalibration.csv --calibration_method conformal_quantile_regression --calibration_path data/delaney3class2.csv --conformal_alpha 0.1