from argparse import ArgumentParser
from multiprocessing import Pool
import sys
from typing import Set
sys.path.append('../')

from tqdm import tqdm

from chemprop.data.utils import get_data
from chemprop.models.jtnn import MolTree


def vocab_for_mol(smiles: str) -> Set[str]:
    mol = MolTree(smiles)
    vocab = {node.smiles for node in mol.nodes}

    return vocab


def generate_vocab(data_path: str, vocab_path: str):
    # Get smiles
    data = get_data(data_path)
    smiles = data.smiles()

    # Create and save vocab
    all_vocab = set()
    with open(vocab_path, 'w') as f:
        for vocab in tqdm(Pool().imap(vocab_for_mol, smiles), total=len(smiles)):
            new_vocab = vocab - all_vocab
            for v in new_vocab:
                f.write(v + '\n')
            all_vocab |= new_vocab


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to data file')
    parser.add_argument('--vocab_path', type=str, required=True,
                        help='Path where vocab will be saved')
    args = parser.parse_args()

    generate_vocab(args.data_path, args.vocab_path)
