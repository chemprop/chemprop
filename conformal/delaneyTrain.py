arguments = [
    '--data_path', 'data/delaney.csv',
    '--dataset_type', 'regression',
    '--save_dir', 'delaney_checkpoints2'
]
if __name__ ==  '__main__':
    import sys
    sys.path.append('../')
    import chemprop
    args = chemprop.args.TrainArgs().parse_args(arguments)
    mean_score, std_score = chemprop.train.cross_validate(args=args, train_func=chemprop.train.run_training)


#for predicts:
#python3 ../predict.py --test_path data/delaney.csv --checkpoint_dir delaney_checkpoints2 --preds_path delaney_preds.csv

#python3 ../predict.py --test_path data/delaney.csv --checkpoint_dir delaney_checkpoints2 --preds_path delaney_preds_dropout.csv --uncertainty_method dropout


#Failed: (or just took too long?)
#python3 ../predict.py --test_path data/delaney.csv --calibration_path data/delaney.csv --checkpoint_dir delaney_checkpoints2 --preds_path delaney_preds_dropout_calib.csv --uncertainty_method dropout --calibration_method zscaling


#Took 36 minutes:
#Maybe its cuz the calibration set was too large lmao!
#python3 ../predict.py --test_path data/delaney.csv --calibration_path data/delaney.csv --checkpoint_dir delaney_checkpoints2 --preds_path delaney_preds_dropout_calib.csv --uncertainty_method dropout

#python3 ../predict.py --test_path data/delaney.csv --calibration_path data/delaneysmall.csv --checkpoint_dir delaney_checkpoints2 --preds_path delaney_preds_dropout_calibsmall.csv --uncertainty_method dropout


#Part 4 but for regression:
#python3 ../predict.py --test_path data/delaney.csv --checkpoint_dir delaney_checkpoints2 --preds_path delaney_preds_conformal.csv --calibration_method conformal_regression --calibration_path data/delaneysmall.csv
#python3 ../predict.py --test_path data/delaney.csv --checkpoint_dir delaney_checkpoints2 --preds_path delaney_preds_conformal.csv --uncertainty_method dropout --calibration_method conformal_regression --calibration_path data/delaneysmall.csv
#it says we need an uncertainty method??

#5.23 Meeting:
#python3 ../predict.py --test_path data/delaney.csv --checkpoint_dir delaney_checkpoints2 --preds_path delaney_preds_conformal.csv --uncertainty_method dropout --calibration_method conformal_regression --calibration_path data/delaneysmall.csv
#Now it should work fine in the save function!