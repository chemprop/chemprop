import sys
sys.path.append('../')


import chemprop
import torch

smiles = [['CCC'], ['CCCC'], ['OCC']]
arguments = [
    '--test_path', '/dev/null',
    '--preds_path', '/dev/null',
    '--checkpoint_dir', 'tox21_checkpoints'
]

args = chemprop.args.PredictArgs().parse_args(arguments)
#preds = chemprop.train.make_predictions(args=args, smiles=smiles)


