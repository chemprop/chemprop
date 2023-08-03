import threading
from collections import OrderedDict
from random import Random
from typing import Dict, Iterator, List, Optional, Union, Tuple

import numpy as np
from torch.utils.data import DataLoader, Dataset, Sampler
from rdkit import Chem

from sklearn.preprocessing import StandardScaler
# from chemprop.v2.featurizers import get_features_generator
from chemprop.v2.featurizers import BatchMolGraph, MolGraph
# from chemprop.v2.featurizers import is_explicit_h, is_reaction, is_adding_hs, is_mol
from chemprop.rdkit import make_mol

from .datasets import MoleculeDataset
from .datapoints import MoleculeDatapoint

# Cache of graph featurizations
CACHE_GRAPH = True
SMILES_TO_GRAPH: Dict[str, MolGraph] = {}


def cache_graph() -> bool:
    r"""Returns whether :class:`~chemprop.features.MolGraph`\ s will be cached."""
    return CACHE_GRAPH


def set_cache_graph(cache_graph: bool) -> None:
    r"""Sets whether :class:`~chemprop.features.MolGraph`\ s will be cached."""
    global CACHE_GRAPH
    CACHE_GRAPH = cache_graph


def empty_cache():
    r"""Empties the cache of :class:`~chemprop.features.MolGraph` and RDKit molecules."""
    SMILES_TO_GRAPH.clear()
    SMILES_TO_MOL.clear()


# Cache of RDKit molecules
CACHE_MOL = True
SMILES_TO_MOL: Dict[str, Union[Chem.Mol, Tuple[Chem.Mol, Chem.Mol]]] = {}


def cache_mol() -> bool:
    r"""Returns whether RDKit molecules will be cached."""
    return CACHE_MOL


def set_cache_mol(cache_mol: bool) -> None:
    r"""Sets whether RDKit molecules will be cached."""
    global CACHE_MOL
    CACHE_MOL = cache_mol

def construct_molecule_batch(data: List[MoleculeDatapoint]) -> MoleculeDataset:
    r"""
    Constructs a :class:`MoleculeDataset` from a list of :class:`MoleculeDatapoint`\ s.

    Additionally, precomputes the :class:`~chemprop.features.BatchMolGraph` for the constructed
    :class:`MoleculeDataset`.

    :param data: A list of :class:`MoleculeDatapoint`\ s.
    :return: A :class:`MoleculeDataset` containing all the :class:`MoleculeDatapoint`\ s.
    """
    data = MoleculeDataset(data)
    data.batch_graph()  # Forces computation and caching of the BatchMolGraph for the molecules

    return data
    
def make_mols(smiles: List[str], reaction_list: List[bool], keep_h_list: List[bool], add_h_list: List[bool]):
    """
    Builds a list of RDKit molecules (or a list of tuples of molecules if reaction is True) for a list of smiles.

    :param smiles: List of SMILES strings.
    :param reaction_list: List of booleans whether the SMILES strings are to be treated as a reaction.
    :param keep_h_list: List of booleans whether to keep hydrogens in the input smiles. This does not add hydrogens, it only keeps them if they are specified.
    :param add_h_list: List of booleasn whether to add hydrogens to the input smiles.
    :return: List of RDKit molecules or list of tuple of molecules.
    """
    mol = []
    for s, reaction, keep_h, add_h in zip(smiles, reaction_list, keep_h_list, add_h_list):
        if reaction:
            mol.append(SMILES_TO_MOL[s] if s in SMILES_TO_MOL else (make_mol(s.split(">")[0], keep_h, add_h), make_mol(s.split(">")[-1], keep_h, add_h)))
        else:
            mol.append(SMILES_TO_MOL[s] if s in SMILES_TO_MOL else make_mol(s, keep_h, add_h))
    return mol
