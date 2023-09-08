Python Usage
============

Model training and predicting can also be embedded within a python script. To train a model, provide arguments as a list of strings (arguments are identical to command line mode),
parse the arguments, and then call :code:`chemprop.train.cross_validate()`::

    import chemprop

    arguments = [
        '--data_path', 'data/tox21.csv',
        '--dataset_type', 'classification',
        '--save_dir', 'tox21_checkpoints'
    ]

    args = chemprop.args.TrainArgs().parse_args(arguments)
    mean_score, std_score = chemprop.train.cross_validate(args=args, train_func=chemprop.train.run_training)

For predicting with a given model, either a list of smiles or a csv file can be used as input. To use a csv file ::

  import chemprop

  arguments = [
      '--test_path', 'data/tox21.csv',
      '--preds_path', 'tox21_preds.csv',
      '--checkpoint_dir', 'tox21_checkpoints'
  ]
  
  args = chemprop.args.PredictArgs().parse_args(arguments)
  preds = chemprop.train.make_predictions(args=args)

If you only want to use the predictions :code:`preds` within the script, and not save the file, set :code:`preds_path` to :code:`/dev/null`. To predict on a list of smiles, run::

  import chemprop

  smiles = [['CCC'], ['CCCC'], ['OCC']]
  arguments = [
      '--test_path', '/dev/null',
      '--preds_path', '/dev/null',
      '--checkpoint_dir', 'tox21_checkpoints'
  ]

  args = chemprop.args.PredictArgs().parse_args(arguments)
  preds = chemprop.train.make_predictions(args=args, smiles=smiles)

where the given :code:`test_path` will be discarded if a list of smiles is provided. If you want to predict multiple sets of molecules consecutively, it is more efficient to
only load the chemprop model once, and then predict with the preloaded model (instead of loading the model for every prediction)::

  import chemprop

  arguments = [
      '--test_path', '/dev/null',
      '--preds_path', '/dev/null',
      '--checkpoint_dir', 'tox21_checkpoints'
  ]

  args = chemprop.args.PredictArgs().parse_args(arguments)

  model_objects = chemprop.train.load_model(args=args)
  
  smiles = [['CCC'], ['CCCC'], ['OCC']]
  preds = chemprop.train.make_predictions(args=args, smiles=smiles, model_objects=model_objects)

  smiles = [['CCCC'], ['CCCCC'], ['COCC']]
  preds = chemprop.train.make_predictions(args=args, smiles=smiles, model_objects=model_objects)
