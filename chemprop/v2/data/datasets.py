from __future__ import annotations

from typing import Iterable, Optional

import numpy as np
from rdkit import Chem
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset

from chemprop.v2.featurizers import MolGraph, MoleculeFeaturizerBase, MoleculeFeaturizer
from chemprop.v2.data.datapoints import MoleculeDatapoint, ReactionDatapoint
from chemprop.v2.featurizers.reaction import ReactionFeaturizer, ReactionFeaturizerBase

Datum = tuple[MolGraph, np.ndarray, np.ndarray, np.ndarray, float, np.ndarray, np.ndarray]


class MolGraphDatasetBase(Dataset):
    def __getitem__(self, idx: int) -> Datum:
        pass

    def __len__(self) -> int:
        return len(self.data)

    @property
    def _targets(self) -> np.ndarray:
        return np.array([d._targets for d in self.data])

    @property
    def targets(self) -> np.ndarray:
        return np.array([d.targets for d in self.data])

    @targets.setter
    def targets(self, Y: np.ndarray) -> None:
        if not len(self.data) == len(Y):
            raise ValueError(
                "number of molecules and targets must be of same length! "
                f"num molecules: {len(self.data)}, num targets: {len(Y)}"
            )

        for d, y in zip(self.data, Y):
            d.targets = y

    @property
    def features(self) -> Optional[np.ndarray]:
        if len(self.data) > 0 and self.data[0].features is None:
            return None

        return np.array([d.features for d in self.data])

    @features.setter
    def features(self, X: np.ndarray):
        if not len(self.data) == len(X):
            raise ValueError(
                "number of molecules and features must be of same length! "
                f"num molecules: {len(self.data)}, num features: {len(X)}"
            )

        for d, x in zip(self.data, X):
            d.features = x

    @property
    def phase_features(self) -> Optional[np.ndarray]:
        if len(self.data) > 0 and self.data[0].phase_features is None:
            return None

        return np.ndarray([d.phase_features for d in self.data])

    @property
    def data_weights(self) -> np.ndarray:
        return np.array([d.data_weight for d in self.data])

    @property
    def gt_targets(self) -> Optional[np.ndarray]:
        if len(self.data) > 0 and self.data[0].gt_targets is None:
            return None

        return np.array([d.gt_targets for d in self.data])

    @property
    def lt_targets(self) -> Optional[np.ndarray]:
        if len(self.data) > 0 and self.data[0].lt_targets is None:
            return None

        return np.array([d.lt_targets for d in self.data])

    @property
    def num_tasks(self) -> Optional[int]:
        return self.data[0].num_tasks if len(self.data) > 0 else None

    @property
    def features_size(self) -> Optional[int]:
        if len(self.data) > 0 and self.data[0].features is None:
            return None

        return len(self.data[0].features)

    def normalize_targets(self, scaler: Optional[StandardScaler] = None) -> StandardScaler:
        """Normalizes the targets of the dataset using a `StandardScaler`

        The StandardScaler subtracts the mean and divides by the standard deviation for each task
        independently. NOTE: This should only be used for regression datasets.

        Returns
        -------
        StandardScaler
            a scaler fit to the targets.
        """
        targets = np.array([d._targets for d in self.data])
        scaler = (scaler or StandardScaler()).fit(targets)
        scaled_targets = scaler.transform(targets)
        self.targets = scaled_targets

        return scaler

    def reset_features_and_targets(self) -> None:
        """Resets the features (atom, bond, and molecule) and targets to their raw values."""
        for d in self.data:
            d.reset_features_and_targets()


class MoleculeDataset(MolGraphDatasetBase):
    """A `MolgraphDataset` composed of `MoleculeDatapoint`s

    Parameters
    ----------
    data : Iterable[MoleculeDatapoint]
        the data from which to create a dataset
    featurizer : MoleculeFeaturizer
        the featurizer with which to generate MolGraphs of the molecules
    """

    def __init__(
        self, data: Iterable[MoleculeDatapoint], featurizer: Optional[MoleculeFeaturizerBase]
    ):
        self.data = list(data)
        self.featurizer = featurizer or MoleculeFeaturizer()

    def __getitem__(self, idx: int) -> Datum:
        d = self.data[idx]

        return (
            self.featurizer(d.mol, d.atom_features, d.bond_features),
            d.atom_descriptors,
            d.features,
            d.targets,
            d.data_weight,
            d.lt_targets,
            d.gt_targets,
        )

    @property
    def smiles(self) -> list[str]:
        return [d.smi for d in self.data]

    @property
    def mols(self) -> list[Chem.Mol]:
        return [d.mol for d in self.data]

    @property
    def number_of_molecules(self) -> int:
        return 1

    @property
    def atom_features(self) -> Optional[np.ndarray]:
        if len(self.data) > 0 and self.data[0].atom_features is None:
            return None

        return np.array([d.atom_features for d in self.data])

    @atom_features.setter
    def atom_features(self, X: np.ndarray):
        if not len(self.data) == len(X):
            raise ValueError(
                "number of molecules and features must be of same length! "
                f"num molecules: {len(self.data)}, num features: {len(X)}"
            )
        for d, x in zip(self.data, X):
            d.atom_features = x

    @property
    def bond_features(self) -> Optional[np.ndarray]:
        if len(self.data) > 0 and self.data[0].bond_features is None:
            return None

        return np.array([d.bond_features for d in self.data])

    @atom_features.setter
    def atom_features(self, X: np.ndarray):
        if not len(self.data) == len(X):
            raise ValueError(
                "number of molecules and features must be of same length! "
                f"num molecules: {len(self.data)}, num features: {len(X)}"
            )
        for d, x in zip(self.data, X):
            d.atom_features = x

    @property
    def atom_descriptors(self) -> Optional[np.ndarray]:
        if len(self.data) > 0 and self.data[0].atom_descriptors is None:
            return None

        return np.array([d.atom_descriptors for d in self.data])

    @atom_features.setter
    def atom_features(self, X: np.ndarray):
        if not len(self.data) == len(X):
            raise ValueError(
                "number of molecules and descriptors must be of same length! "
                f"num molecules: {len(self.data)}, num descriptors: {len(X)}"
            )
        for d, x in zip(self.data, X):
            d.atom_features = x

    @property
    def atom_features_size(self) -> Optional[int]:
        if len(self.data) > 0 and self.data[0].atom_features is None:
            return None

        return len(self.data[0].atom_features[0])

    @property
    def bond_features_size(self) -> Optional[int]:
        if len(self.data) > 0 and self.data[0].bond_features is None:
            return None

        return len(self.data[0].bond_features[0])

    @property
    def atom_descriptors_size(self) -> Optional[int]:
        if len(self.data) > 0 and self.data[0].atom_descriptors is None:
            return None

        return len(self.data[0].atom_descriptors[0])

    def normalize(
        self, key: str = "features", scaler: Optional[StandardScaler] = None
    ) -> StandardScaler:
        valid_keys = ("features", "atom_features", "bond_features", "atom_descriptors", "all")
        if key.lower() not in valid_keys:
            raise ValueError(f"Invalid feature key! got: {key}. expected one of: {valid_keys}")

        if key == "features":
            X = self.features
        elif key == "atom_features":
            X = self.atom_features
        elif key == "bond_features":
            X = self.bond_features
        elif key == "atom_descriptors":
            X = self.atom_descriptors
        else:
            return [self.normalize(k, scaler) for k in valid_keys if k != "all"]

        if X is None:
            return scaler

        if scaler is None:
            scaler = StandardScaler().fit(X)
        X_normalized = scaler.transform(X)

        if key == "features":
            self.features = X_normalized
        elif key == "atom_features":
            self.atom_features = X_normalized
        elif key == "bond_features":
            self.bond_features = X_normalized
        elif key == "atom_descriptors":
            self.atom_descriptors = X_normalized

        return scaler


class ReactionDataset(MolGraphDatasetBase):
    """A `MolgraphDataset` composed of `ReactionDatapoint`s

    Parameters
    ----------
    data : Iterable[ReactionDatapoint]
        the dataset from which to load
    featurizer : ReactionFeaturizer
        the featurizer with which to generate MolGraphs of the input
    """

    def __init__(
        self, data: Iterable[ReactionDatapoint], featurizer: Optional[ReactionFeaturizerBase]
    ):
        self.data = list(data)
        self.featurizer = featurizer or ReactionFeaturizer()

    def __getitem__(self, idx: int) -> Datum:
        d = self.data[idx]

        return (
            self.featurizer(d.mols, d.atom_features, d.bond_features),
            None,
            d.features,
            d.targets,
            d.data_weight,
            d.lt_targets,
            d.gt_targets,
        )

    @property
    def smiles(self) -> list[str]:
        return [d.smis for d in self.data]

    @property
    def mols(self) -> list[Chem.Mol]:
        return [d.mols for d in self.data]

    @property
    def number_of_molecules(self) -> int:
        return self.data[0].number_of_molecules
