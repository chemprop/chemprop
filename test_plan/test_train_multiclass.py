arguments = [
    '--data_path', 'data/tox21_3class1.csv',
    '--dataset_type', 'multiclass',
    '--save_dir', 'multiclass/tox21_3class1',
    '--multiclass_num_classes',  '10'
]
if __name__ ==  '__main__':
    import sys
    sys.path.append('../')
    import chemprop
    args = chemprop.args.TrainArgs().parse_args(arguments)
    mean_score, std_score = chemprop.train.cross_validate(args=args, train_func=chemprop.train.run_training)


#Prediction:
#python3 ../predict.py --test_path data/tox21smallclass.csv --checkpoint_dir tox21_class_checkpoints --preds_path tox21_preds_multiclass.csv
#python3 ../predict.py --test_path data/tox21smallclass.csv --checkpoint_dir tox21_class_checkpoints --preds_path tox21_preds_conformal_multiclass.csv --calibration_method conformal --calibration_path data/tox21smallclass.csv
#python3 ../predict.py --test_path data/tox21smallclass.csv --checkpoint_dir tox21_class_checkpoints --preds_path tox21_preds_conformal_adaptive_multiclass.csv --calibration_method conformal_adaptive --calibration_path data/tox21smallclass.csv