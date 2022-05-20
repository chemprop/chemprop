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




