from argparse import ArgumentParser, Namespace
import os

import torch
from tqdm import trange

from chemprop.data.utils import get_data
from chemprop.utils import load_checkpoint


def visualize_attention(args: Namespace):
    """Visualizes attention weights."""
    print('Loading data')
    data = get_data(args.data_path)
    smiles = data.smiles()
    print('Data size = {:,}'.format(len(smiles)))

    print('Loading model from "{}"'.format(args.checkpoint_path))
    model, _, _, _ = load_checkpoint(args.checkpoint_path, cuda=args.cuda)
    mpn = model[0]

    for i in trange(0, len(smiles), args.batch_size):
        smiles_batch = smiles[i:i + args.batch_size]
        mpn.viz_attention(smiles_batch, viz_dir=args.viz_dir)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to data CSV file')
    parser.add_argument('--viz_dir', type=str, required=True,
                        help='Path where attention PNGs will be saved')
    parser.add_argument('--checkpoint_path', type=str, required=True,
                        help='Path to a model checkpoint')
    parser.add_argument('--batch_size', type=int, default=50,
                        help='Batch size')
    parser.add_argument('--no_cuda', action='store_true', default=False,
                        help='Turn off cuda')
    args = parser.parse_args()

    # Cuda
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    del args.no_cuda

    # Create directory for preds path
    os.makedirs(args.viz_dir, exist_ok=True)

    visualize_attention(args)
