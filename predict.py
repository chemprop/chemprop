from argparse import ArgumentParser
import os

import torch

from chemprop.parsing import add_predict_args, update_args_from_checkpoint_dir
from chemprop.train import make_predictions

if __name__ == '__main__':
    parser = ArgumentParser()
    add_predict_args(parser)
    parser.add_argument('--checkpoint_dir', type=str, required=True,
                        help='Directory from which to load model checkpoints'
                             '(walks directory and ensembles all models that are found)')
    parser.add_argument('--batch_size', type=int, default=50,
                        help='Batch size')
    parser.add_argument('--no_cuda', action='store_true', default=False,
                        help='Turn off cuda')
    args = parser.parse_args()

    # Get checkpoint paths
    update_args_from_checkpoint_dir(args)

    # Cuda
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    del args.no_cuda

    # Create directory for preds path
    preds_dir = os.path.dirname(args.preds_path)
    if preds_dir != '':
        os.makedirs(preds_dir, exist_ok=True)

    make_predictions(args)
