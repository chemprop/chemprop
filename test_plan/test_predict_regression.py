import os

alpha_list = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
#alpha_list = [0.1]


for alpha in alpha_list:
    #os.system(f"python3 ../predict.py --test_path data/delaney3class3.csv --checkpoint_dir delaney_checkpoints_3class1 --preds_path regression_new/delaney_preds_conformal_{alpha}.csv --calibration_method conformal_regression --calibration_path data/delaney3class2.csv --alpha {alpha}")
    os.system(f"python3 ../predict.py --test_path data/delaney3class3.csv --checkpoint_dir delaney_checkpoints_3class1 --preds_path regression/delaney_preds_conformal_{alpha}.csv --calibration_method conformal_regression --calibration_path data/delaney3class2.csv --conformal_alpha {alpha} --evaluation_methods conformal_coverage --evaluation_scores_path regression/delaney_eval_conformal_{alpha}.csv")

