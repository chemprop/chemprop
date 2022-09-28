arguments = [
    '--data_path', 'data/tox21_multilabel1.csv',
    '--dataset_type', 'classification',
    '--save_dir', 'multilabel/tox21_multilabel1',
]
if __name__ ==  '__main__':
    import sys
    sys.path.append('../')
    import chemprop
    args = chemprop.args.TrainArgs().parse_args(arguments)
    mean_score, std_score = chemprop.train.cross_validate(args=args, train_func=chemprop.train.run_training)

