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