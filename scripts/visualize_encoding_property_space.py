from argparse import ArgumentParser, Namespace
import os
import random
import sys
sys.path.append('../')
from typing import List

import numpy as np
import ternary
from tqdm import trange

from chemprop.data import MoleculeDataset
from chemprop.data.utils import get_data
from chemprop.utils import load_checkpoint, load_scalers


def visualize_encoding_property_space(args: Namespace):
    # Load data
    data = get_data(args.data_path)

    # Sort according to similarity measure
    if args.similarity_measure == 'property':
        data.sort(key=lambda d: d.targets[args.task_index])
    elif args.similarity_measure == 'random':
        data.shuffle(args.seed)
    else:
        raise ValueError('similarity_measure "{}" not supported or not implemented yet.'.format(args.similarity_measure))

    # Load model and scalers
    model = load_checkpoint(args.checkpoint_path)
    scaler, features_scaler = load_scalers(args.checkpoint_path)
    data.normalize_features(features_scaler)

    # Random seed
    if args.seed is not None:
        random.seed(args.seed)

    # Generate visualizations
    for i in trange(args.num_examples):
        # Get random three molecules with similar properties
        index = random.randint(1, len(data) - 2)
        molecules = MoleculeDataset(data[index - 1:index + 2])
        molecule_targets = [t[args.task_index] for t in molecules.targets()]

        # Encode three molecules
        molecule_encodings = model.encoder(molecules.smiles())

        # Define interpolation
        def predict_property(point: List[int]) -> float:
            # Return true value on endpoints of triangle
            argmax = np.argmax(point)
            if point[argmax] == 1:
                return molecule_targets[argmax]

            # Interpolate and predict task value
            encoding = sum(point[j] * molecule_encodings[j] for j in range(len(molecule_encodings)))
            pred = model.ffn(encoding).data.cpu().numpy()
            pred = scaler.inverse_transform(pred)
            pred = pred.item()

            return pred

        # Create visualization
        scale = 20
        fontsize = 6

        figure, tax = ternary.figure(scale=scale)
        tax.heatmapf(predict_property, boundary=True, style="hexagonal")
        tax.set_title("Property Prediction")
        tax.right_axis_label('{} ({:.3f}) -->'.format(molecules[0].smiles, molecules[0].targets[args.task_index]),
                             fontsize=fontsize)
        tax.left_axis_label('{} ({:.3f}) -->'.format(molecules[1].smiles, molecules[1].targets[args.task_index]),
                            fontsize=fontsize)
        tax.bottom_axis_label('<-- {} ({:.3f})'.format(molecules[2].smiles, molecules[2].targets[args.task_index]),
                              fontsize=fontsize)

        tax.savefig(os.path.join(args.save_dir, '{}.png'.format(i)))


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to regression dataset .csv')
    parser.add_argument('--checkpoint_path', type=str, required=True,
                        help='Path to a model checkpoint .pt file')
    parser.add_argument('--similarity_measure', type=str, default='random',
                        choices=['random', 'random'],
                        help='Similarity measure to use when choosing the three molecules for each visualization')
    parser.add_argument('--task_index', type=int, default=0,
                        help='Index of the task (property) in the dataset to use')
    parser.add_argument('--num_examples', type=int, default=10,
                        help='Number of visualizations to generate')
    parser.add_argument('--save_dir', type=str, required=True,
                        help='Directory where visualizations will be saved')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for choosing three similar molecules')
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    visualize_encoding_property_space(args)
