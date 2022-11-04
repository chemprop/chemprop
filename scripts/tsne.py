import os
import sys
import time
from typing import List

from matplotlib import offsetbox
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from rdkit import Chem
from rdkit.Chem import Draw
from sklearn.manifold import TSNE
from tap import Tap
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from chemprop.data import get_smiles
from chemprop.features import get_features_generator
from chemprop.utils import makedirs


class Args(Tap):
    smiles_paths: List[str]  # Path to .csv files containing smiles strings (with header)
    smiles_column: str = None  # Name of the column containing SMILES strings for the first data. By default, uses the first column.
    colors: List[str] = ['red', 'green', 'orange', 'purple', 'blue']  # Colors of the points associated with each dataset
    sizes: List[float] = [1, 1, 1, 1, 1]  # Sizes of the points associated with each molecule
    scale: int = 1  # Scale of figure
    plot_molecules: bool = False  # Whether to plot images of molecules instead of points
    max_per_dataset: int = 10000  # Maximum number of molecules per dataset; larger datasets will be subsampled to this size
    save_path: str  # Path to a .png file where the t-SNE plot will be saved
    cluster: bool = False  # Whether to create new clusters from all smiles, ignoring original csv groupings


def compare_datasets_tsne(args: Args):
    if len(args.smiles_paths) > len(args.colors) or len(args.smiles_paths) > len(args.sizes):
        raise ValueError('Must have at least as many colors and sizes as datasets')

    # Random seed for random subsampling
    np.random.seed(0)

    # Load the smiles datasets
    print('Loading data')
    smiles, slices, labels = [], [], []
    for smiles_path in args.smiles_paths:
        # Get label
        label = os.path.basename(smiles_path).replace('.csv', '')

        # Get SMILES
        new_smiles = get_smiles(path=smiles_path, smiles_columns=args.smiles_column, flatten=True)
        print(f'{label}: {len(new_smiles):,}')

        # Subsample if dataset is too large
        if len(new_smiles) > args.max_per_dataset:
            print(f'Subsampling to {args.max_per_dataset:,} molecules')
            new_smiles = np.random.choice(new_smiles, size=args.max_per_dataset, replace=False).tolist()

        slices.append(slice(len(smiles), len(smiles) + len(new_smiles)))
        labels.append(label)
        smiles += new_smiles

    # Compute Morgan fingerprints
    print('Computing Morgan fingerprints')
    morgan_generator = get_features_generator('morgan')
    morgans = [morgan_generator(smile) for smile in tqdm(smiles, total=len(smiles))]

    print('Running t-SNE')
    start = time.time()
    tsne = TSNE(n_components=2, init='pca', random_state=0, metric='jaccard')
    X = tsne.fit_transform(morgans)
    print(f'time = {time.time() - start:.2f} seconds')

    if args.cluster:
        import hdbscan  # pip install hdbscan
        print('Running HDBSCAN')
        start = time.time()
        clusterer = hdbscan.HDBSCAN(min_cluster_size=5, gen_min_span_tree=True)
        colors = clusterer.fit_predict(X)
        print(f'time = {time.time() - start:.2f} seconds')

    print('Plotting t-SNE')
    x_min, x_max = np.min(X, axis=0), np.max(X, axis=0)
    X = (X - x_min) / (x_max - x_min)

    makedirs(args.save_path, isfile=True)

    plt.clf()
    fontsize = 50 * args.scale
    fig = plt.figure(figsize=(64 * args.scale, 48 * args.scale))
    plt.title('t-SNE using Morgan fingerprint with Jaccard similarity', fontsize=2 * fontsize)
    ax = fig.gca()
    handles = []
    legend_kwargs = dict(loc='upper right', fontsize=fontsize)

    if args.cluster:
        plt.scatter(X[:, 0], X[:, 1], s=150 * np.mean(args.sizes), c=colors, cmap='nipy_spectral')
    else:
        for slc, color, label, size in zip(slices, args.colors, labels, args.sizes):
            if args.plot_molecules:
                # Plots molecules
                handles.append(mpatches.Patch(color=color, label=label))

                for smile, (x, y) in zip(smiles[slc], X[slc]):
                    img = Draw.MolsToGridImage([Chem.MolFromSmiles(smile)], molsPerRow=1, subImgSize=(200, 200))
                    imagebox = offsetbox.AnnotationBbox(offsetbox.OffsetImage(img), (x, y), bboxprops=dict(color=color))
                    ax.add_artist(imagebox)
            else:
                # Plots points
                plt.scatter(X[slc, 0], X[slc, 1], s=150 * size, color=color, label=label)

        if args.plot_molecules:
            legend_kwargs['handles'] = handles

    plt.legend(**legend_kwargs)
    plt.xticks([]), plt.yticks([])

    print('Saving t-SNE')
    plt.savefig(args.save_path)


if __name__ == '__main__':
    compare_datasets_tsne(Args().parse_args())
