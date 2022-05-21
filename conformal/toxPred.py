import sys
sys.path.append('../')


import chemprop

arguments = [
    '--test_path', 'data/tox21.csv',
    '--preds_path', 'tox21_preds.csv',
    '--checkpoint_dir', 'tox21_checkpoints'
]

args = chemprop.args.PredictArgs().parse_args(arguments)
preds = chemprop.train.make_predictions(args=args)

#Train takes 5 minutes


#for predicts:
#python3 ../predict.py --test_path data/tox21.csv --checkpoint_dir tox21_checkpoints2 --preds_path tox21_preds2.csv
#Takes 1:25

#python3 ../predict.py --test_path data/tox21.csv --checkpoint_dir tox21_checkpoints2 --preds_path tox21_preds_conformal.csv --uncertainty_method conformal

#python3 ../predict.py --test_path data/tox21.csv --checkpoint_dir tox21_checkpoints2 --preds_path tox21_preds_conformal_adaptive.csv --uncertainty_method conformal_adaptive


#uncertainty method(prediction) but no calibration
#python3 ../predict.py --test_path data/tox21.csv --checkpoint_dir tox21_checkpoints2 --preds_path tox21_preds_dropout.csv --uncertainty_method dropout


#PART 2:
#Calibrator with hardcoded qhat value. No calibration path. No uncertainty method
#python3 ../predict.py --test_path data/tox21.csv --checkpoint_dir tox21_checkpoints2 --preds_path tox21_preds_conformal_part_2.csv --calibration_method conformal
#Took 1:07

#I think we need calibration path in order for calibrator to even be initialized??
#python3 ../predict.py --test_path data/tox21.csv --checkpoint_dir tox21_checkpoints2 --preds_path tox21_preds_conformal_part_2.csv --calibration_method conformal --calibration_path data/tox21small.csv
#Took 1:49

