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

#Part 4:
#python3 ../predict.py --test_path data/tox21.csv --checkpoint_dir tox21_checkpoints2 --preds_path tox21_preds_conformal_part_4.csv --calibration_method conformal --calibration_path data/tox21small.csv
#By default, alpha is 0.1
#alpha is the error rate

#python3 ../predict.py --test_path data/tox21.csv --checkpoint_dir tox21_checkpoints2 --preds_path tox21_preds_conformal_part_4_alpha05.csv --calibration_method conformal --calibration_path data/tox21small.csv --alpha 0.5
#Use command line args to change alpha to 0.5

#Part 4 finished! The problem with tox21 dataset is that its multiclass, and some datapoints have no class which messess stuff up!


#Adaptive test:
#python3 ../predict.py --test_path data/tox21.csv --checkpoint_dir tox21_checkpoints2 --preds_path tox21_preds_conformal_adaptive.csv --calibration_method conformal_adaptive --calibration_path data/tox21small.csv