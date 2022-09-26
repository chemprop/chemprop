arguments = [
    '--data_path', 'data/delaney3class1.csv',
    '--dataset_type', 'regression',
    '--save_dir', 'delaney_checkpoints_3class1'
]


if __name__ ==  '__main__':
    import sys
    sys.path.append('../')
    import chemprop
    args = chemprop.args.TrainArgs().parse_args(arguments)
    mean_score, std_score = chemprop.train.cross_validate(args=args, train_func=chemprop.train.run_training)