from collections import OrderedDict
from functools import partial
from random import Random
from typing import Callable, Dict, Iterator, List, Union

import numpy as np
from torch.utils.data import DataLoader, Dataset, Sampler
from rdkit import Chem

from .scaler import StandardScaler
from chemprop.features import get_features_generator
from chemprop.features import BatchMolGraph, MolGraph


# Cache of graph featurizations
SMILES_TO_GRAPH: Dict[str, MolGraph] = {}


class MoleculeDatapoint:
    """A MoleculeDatapoint contains a single molecule and its associated features and targets."""

    def __init__(self,
                 smiles: str,
                 targets: List[float] = None,
                 row: OrderedDict = None,
                 features: np.ndarray = None,
                 features_generator: List[str] = None):
        """
        Initializes a MoleculeDatapoint, which contains a single molecule.

        ;param smiles: The SMILES string for the molecule.
        :param targets: A list of targets for the molecule (contains None for unknown target values).
        :param row: The raw CSV row containing the information for this molecule.
        :param args: Arguments.
        :param features: A numpy array containing additional features (ex. Morgan fingerprint).
        """
        if features is not None and features_generator is not None:
            raise ValueError('Cannot provide both loaded features and a features generator.')

        self.smiles = smiles
        self.targets = targets
        self.row = row or OrderedDict()
        self.features = features
        self.features_generator = features_generator
        self._mol = 'None'  # Initialize with 'None' to distinguish between None returned by invalid molecule

        # Generate additional features if given a generator
        if self.features_generator is not None:
            self.features = []

            for fg in self.features_generator:
                features_generator = get_features_generator(fg)
                if self.mol is not None and self.mol.GetNumHeavyAtoms() > 0:
                    self.features.extend(features_generator(self.mol))

            self.features = np.array(self.features)

        # Fix nans in features
        if self.features is not None:
            replace_token = 0
            self.features = np.where(np.isnan(self.features), replace_token, self.features)

    @property
    def mol(self) -> Chem.Mol:
        """Get the RDKit molecule for the SMILES string (with lazy loading)."""
        if self._mol == 'None':
            self._mol = Chem.MolFromSmiles(self.smiles)

        return self._mol

    def set_features(self, features: np.ndarray):
        """
        Sets the features of the molecule.

        :param features: A 1-D numpy array of features for the molecule.
        """
        self.features = features

    def num_tasks(self) -> int:
        """
        Returns the number of prediction tasks.

        :return: The number of tasks.
        """
        return len(self.targets)

    def set_targets(self, targets: List[float]):
        """
        Sets the targets of a molecule.

        :param targets: A list of floats containing the targets.
        """
        self.targets = targets


class MoleculeDataset(Dataset):
    """A MoleculeDataset contains a list of molecules and their associated features and targets."""

    def __init__(self, data: List[MoleculeDatapoint]):
        """
        Initializes a MoleculeDataset, which contains a list of MoleculeDatapoints (i.e. a list of molecules).

        :param data: A list of MoleculeDatapoints.
        """
        self._data = data
        self._scaler = None
        self._batch_graph = None
        self._random = Random()

    def smiles(self) -> List[str]:
        """
        Returns the smiles strings associated with the molecules.

        :return: A list of smiles strings.
        """
        return [d.smiles for d in self._data]

    def mols(self) -> List[Chem.Mol]:
        """
        Returns the RDKit molecules associated with the molecules.

        :return: A list of RDKit Mols.
        """
        return [d.mol for d in self._data]

    def batch_graph(self, cache: bool = False) -> BatchMolGraph:
        """
        Returns a BatchMolGraph with the graph featurization of the molecules.

        :param cache: Whether to store the graph featurizations in the global cache.
        :return: A BatchMolGraph.
        """
        if self._batch_graph is None:
            mol_graphs = []
            for d in self._data:
                if d.smiles in SMILES_TO_GRAPH:
                    mol_graph = SMILES_TO_GRAPH[d.smiles]
                else:
                    mol_graph = MolGraph(d.mol)
                    if cache:
                        SMILES_TO_GRAPH[d.smiles] = mol_graph
                mol_graphs.append(mol_graph)

            self._batch_graph = BatchMolGraph(mol_graphs)

        return self._batch_graph

    def features(self) -> List[np.ndarray]:
        """
        Returns the features associated with each molecule (if they exist).

        :return: A list of 1D numpy arrays containing the features for each molecule or None if there are no features.
        """
        if len(self._data) == 0 or self._data[0].features is None:
            return None

        return [d.features for d in self._data]

    def targets(self) -> List[List[float]]:
        """
        Returns the targets associated with each molecule.

        :return: A list of lists of floats containing the targets.
        """
        return [d.targets for d in self._data]

    def num_tasks(self) -> int:
        """
        Returns the number of prediction tasks.

        :return: The number of tasks.
        """
        return self._data[0].num_tasks() if len(self._data) > 0 else None

    def features_size(self) -> int:
        """
        Returns the size of the features array associated with each molecule.

        :return: The size of the features.
        """
        return len(self._data[0].features) if len(self._data) > 0 and self._data[0].features is not None else None

    def shuffle(self, seed: int = None):
        """
        Shuffles the dataset.

        :param seed: Optional random seed.
        """
        if seed is not None:
            self._random.seed(seed)
        self._random.shuffle(self._data)
    
    def normalize_features(self, scaler: StandardScaler = None, replace_nan_token: int = 0) -> StandardScaler:
        """
        Normalizes the features of the dataset using a StandardScaler (subtract mean, divide by standard deviation).

        If a scaler is provided, uses that scaler to perform the normalization. Otherwise fits a scaler to the
        features in the dataset and then performs the normalization.

        :param scaler: A fitted StandardScaler. Used if provided. Otherwise a StandardScaler is fit on
        this dataset and is then used.
        :param replace_nan_token: What to replace nans with.
        :return: A fitted StandardScaler. If a scaler is provided, this is the same scaler. Otherwise, this is
        a scaler fit on this dataset.
        """
        if len(self._data) == 0 or self._data[0].features is None:
            return None

        if scaler is not None:
            self._scaler = scaler

        elif self._scaler is None:
            features = np.vstack([d.features for d in self._data])
            self._scaler = StandardScaler(replace_nan_token=replace_nan_token)
            self._scaler.fit(features)

        for d in self._data:
            d.set_features(self._scaler.transform(d.features.reshape(1, -1))[0])

        return self._scaler
    
    def set_targets(self, targets: List[List[float]]):
        """
        Sets the targets for each molecule in the dataset. Assumes the targets are aligned with the datapoints.

        :param targets: A list of lists of floats containing targets for each molecule. This must be the
        same length as the underlying dataset.
        """
        assert len(self._data) == len(targets)
        for i in range(len(self._data)):
            self._data[i].set_targets(targets[i])

    def sort(self, key: Callable):
        """
        Sorts the dataset using the provided key.

        :param key: A function on a MoleculeDatapoint to determine the sorting order.
        """
        self._data.sort(key=key)

    def __len__(self) -> int:
        """
        Returns the length of the dataset (i.e. the number of molecules).

        :return: The length of the dataset.
        """
        return len(self._data)

    def __getitem__(self, item) -> Union[MoleculeDatapoint, List[MoleculeDatapoint]]:
        """
        Gets one or more MoleculeDatapoints via an index or slice.

        :param item: An index (int) or a slice object.
        :return: A MoleculeDatapoint if an int is provided or a list of MoleculeDatapoints if a slice is provided.
        """
        return self._data[item]


class MoleculeSampler(Sampler):
    """A Sampler for MoleculeDataLoaders."""

    def __init__(self,
                 dataset: MoleculeDataset,
                 class_balance: bool = False,
                 shuffle: bool = False,
                 seed: int = 0):
        """
        Initializes the MoleculeSampler.

        :param class_balance: Whether to perform class balancing (i.e. use an equal number of positive and negative molecules).
                              Class balance is only available for single task classification datasets.
                              Set shuffle to True in order to get a random subset of the larger class.
        :param shuffle: Whether to shuffle the data.
        :param seed: Random seed. Only needed if shuffle is True.
        """
        super(Sampler, self).__init__()

        self.dataset = dataset
        self.class_balance = class_balance
        self.shuffle = shuffle

        self._random = Random(seed)

        if self.class_balance:
            if self.dataset.num_tasks() != 1:
                raise ValueError('Class balance can only be used on single-task classification datasets.')

            self.positive_indices = [index for index, datapoint in enumerate(dataset) if datapoint.targets[0] == 1]
            self.negative_indices = [index for index, datapoint in enumerate(dataset) if datapoint.targets[0] == 0]

            self.length = 2 * min(len(self.positive_indices), len(self.negative_indices))
        else:
            self.positive_indices = self.negative_indices = None

            self.length = len(self.dataset)

    def __iter__(self) -> Iterator[int]:
        """Creates an iterator over indices to sample."""
        if self.class_balance:
            if self.shuffle:
                self._random.shuffle(self.positive_indices)
                self._random.shuffle(self.negative_indices)

            indices = [index for pair in zip(self.positive_indices, self.negative_indices) for index in pair]
        else:
            indices = list(range(len(self.dataset)))

            if self.shuffle:
                self._random.shuffle(indices)

        return iter(indices)

    def __len__(self) -> int:
        """Returns the number of indices that will be sampled."""
        return self.length


def construct_molecule_batch(data: List[MoleculeDatapoint], cache: bool = False) -> MoleculeDataset:
    """
    Constructs a MoleculeDataset from a list of MoleculeDatapoints while also constructing the BatchMolGraph.

    :param data: A list of MoleculeDatapoints.
    :param cache: Whether to cache the graph featurizations of molecules for faster processing.
    :return: A MoleculeDataset with all the MoleculeDatapoints and a BatchMolGraph graph featurization.
    """
    data = MoleculeDataset(data)
    data.batch_graph(cache=cache)  # Forces computation and caching of the BatchMolGraph for the molecules

    return data


class MoleculeDataLoader(DataLoader):
    """A DataLoader for MoleculeDatasets."""

    def __init__(self,
                 dataset: MoleculeDataset,
                 batch_size: int = 50,
                 num_workers: int = 8,
                 cache: bool = False,
                 class_balance: bool = False,
                 shuffle: bool = False,
                 seed: int = 0):
        """
        Initializes the MoleculeDataLoader.

        :param dataset: The MoleculeDataset containing the molecules.
        :param batch_size: Batch size.
        :param num_workers: Number of workers used to build batches.
        :param cache: Whether to cache the graph featurizations of molecules for faster processing.
        :param class_balance: Whether to perform class balancing (i.e. use an equal number of positive and negative molecules).
                              Class balance is only available for single task classification datasets.
                              Set shuffle to True in order to get a random subset of the larger class.
        :param shuffle: Whether to shuffle the data.
        :param seed: Random seed. Only needed if shuffle is True.
        """
        self._dataset = dataset
        self._batch_size = batch_size
        self._num_workers = num_workers
        self._cache = cache
        self._class_balance = class_balance
        self._shuffle = shuffle
        self._seed = seed

        self._sampler = MoleculeSampler(
            dataset=self._dataset,
            class_balance=self._class_balance,
            shuffle=self._shuffle,
            seed=self._seed
        )

        super(MoleculeDataLoader, self).__init__(
            dataset=self._dataset,
            batch_size=self._batch_size,
            sampler=self._sampler,
            num_workers=self._num_workers,
            collate_fn=partial(construct_molecule_batch, cache=self._cache)
        )

    def targets(self) -> List[List[float]]:
        """
        Returns the targets associated with each molecule.

        :return: A list of lists of floats containing the targets.
        """
        if self._class_balance or self._shuffle:
            raise ValueError('Cannot safely extract targets when class balance or shuffle are enabled.')

        return [self._dataset[index].targets for index in self._sampler]

    def __iter__(self) -> Iterator[MoleculeDataset]:
        return super(MoleculeDataLoader, self).__iter__()
