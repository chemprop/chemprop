arguments = [
    '--data_path', 'data/tox21.csv',
    '--dataset_type', 'classification',
    '--save_dir', 'tox21_checkpoints2'
]
if __name__ ==  '__main__':
    import sys
    sys.path.append('../chemprop/')
    import chemprop
    args = chemprop.args.TrainArgs().parse_args(arguments)
    mean_score, std_score = chemprop.train.cross_validate(args=args, train_func=chemprop.train.run_training)