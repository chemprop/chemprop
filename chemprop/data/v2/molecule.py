from collections import OrderedDict
from dataclasses import InitVar, dataclass
from typing import List, Optional, Sequence

import numpy as np
from rdkit import Chem
from torch.utils.data import Dataset

from chemprop.data.scaler import StandardScaler
from chemprop.featurizers import get_features_generator
from chemprop.featurizers import MolGraph
from chemprop.featurizers.molgraph import MolGraphFeaturizer
from chemprop.rdkit import make_mol

@dataclass
class MoleculeDatapoint:
    """A :class:`MoleculeDatapoint` contains a single molecule and its associated features and targets.
    
    :param smiles: A list of the SMILES strings for the molecules.
    :param targets: the targets for the molecule with `nan` indicating unknown targets
    :param row: The raw CSV row containing the information for this molecule.
    :param data_weight: Weighting of the datapoint for the loss function.
    :param gt_targets: Indicates whether the targets are an inequality regression target of the 
    form ">x"
    :param lt_targets: Indicates whether the targets are an inequality regression target of the 
    form "<x"
    :param features: A numpy array containing additional features (e.g., Morgan fingerprint).
    :param features_generator: A list of features generators to use.
    :param phase_features: A one-hot vector indicating the phase of the data, as used in spectra data.
    :param atom_descriptors: A numpy array containing additional atom descriptors to featurize the molecule
    :param bond_features: A numpy array containing additional bond features to featurize the molecule
    :param overwrite_default_atom_features: Boolean to overwrite default atom features by atom_features
    :param overwrite_default_bond_features: Boolean to overwrite default bond features by bond_features
    """
    smiles: str
    targets: np.ndarray
    row: OrderedDict = None
    data_weight: float = 1
    gt_targets: List[bool] = None
    lt_targets: List[bool] = None
    features: np.ndarray = None
    features_generator: InitVar[Optional[List[str]]] = None
    phase_features: List[float] = None
    atom_features: np.ndarray = None
    atom_descriptors: np.ndarray = None
    bond_features: np.ndarray = None
    overwrite_default_atom_features: bool = False
    overwrite_default_bond_features: bool = False
    explicit_h: bool = False
    add_h: bool = False

    def __post_init__(self, features_generator):
        if self.features is not None and features_generator is not None:
            raise ValueError("Cannot provide both loaded features and a features generator.")

        self.mol = make_mol(self.smiles, self.explicit_h, self.add_h)

        if self.features_generator is not None:
            self.features = []

            for fg in self.features_generator:
                features_generator = get_features_generator(fg)
                m = self.mol
                if m is not None and m.GetNumHeavyAtoms() > 0:
                    self.features.extend(features_generator(m))
                # for H2
                elif m is not None and m.GetNumHeavyAtoms() == 0:
                    # not all features are equally long, so use methane as dummy molecule to determine length
                    self.features.extend(
                        np.zeros(len(features_generator(Chem.MolFromSmiles("C"))))
                    )

            self.features = np.array(self.features)

        replace_token = 0
        if self.features is not None:
            self.features[np.isnan(self.features)] = replace_token

        if self.atom_descriptors is not None:
            self.atom_descriptors[np.isnan(self.atom_descriptors)] = replace_token

        if self.atom_features is not None:
            self.atom_features[np.isnan(self.atom_features)] = replace_token

        if self.bond_features is not None:
            self.bond_features[np.isnan(self.bond_features)] = replace_token

        self.raw_features = self.features
        self.raw_targets = self.targets
        self.raw_atom_descriptors = self.atom_descriptors 
        self.raw_atom_features = self.atom_features
        self.raw_bond_features = self.bond_features


    @property
    def number_of_molecules(self) -> int:
        return 1

    @property
    def num_tasks(self) -> int:
        return len(self.targets)

    def reset_features_and_targets(self) -> None:
        """Resets the features (atom, bond, and molecule) and targets to their raw values."""
        self.features, self.targets = self.raw_features, self.raw_targets
        self.atom_descriptors, self.atom_features, self.bond_features = (
            self.raw_atom_descriptors,
            self.raw_atom_features,
            self.raw_bond_features,
        )


class MolGraphDataset(Dataset):
    r"""A :class:`MoleculeDataset` contains a list of :class:`MoleculeDatapoint`\ s with access to 
    their attributes."""

    def __init__(self, data: Sequence[MoleculeDatapoint], featurizer: MolGraphFeaturizer):
        r"""
        :param data: A list of :class:`MoleculeDatapoint`\ s.
        """
        if data is None:
            raise ValueError("arg: `data` was None!")

        self.data = data
        self.featurizer = featurizer

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> MolGraph:
        d = self.data[idx]

        return self.featurizer(d.smiles, d.atom_features, d.bond_features)

    @property
    def smiles(self) -> list[str]:
        return [d.smiles for d in self.data]

    @property
    def mols(self) -> list[Chem.Mol]:
        return [d.mol for d in self.data]

    @property
    def number_of_molecules(self) -> int:
        return 1

    @property
    def features(self) -> np.ndarray:
        if len(self.data) > 0 and self.data[0].features is None:
            return None

        return np.array([d.features for d in self.data])

    @property
    def phase_features(self) -> list[np.ndarray]:
        if len(self.data) > 0 and self.data[0].phase_features is None:
            return None

        return [d.phase_features for d in self.data]

    @property
    def atom_features(self) -> list[np.ndarray]:
        if len(self.data) > 0 and  self.data[0].atom_features is None:
            return None

        return [d.atom_features for d in self.data]

    @property
    def atom_descriptors(self) -> list[np.ndarray]:
        if len(self.data) > 0 and self.data[0].atom_descriptors is None:
            return None

        return [d.atom_descriptors for d in self.data]

    @property
    def bond_features(self) -> list[np.ndarray]:
        if len(self.data) > 0 and self.data[0].bond_features is None:
            return None

        return [d.bond_features for d in self.data]

    @property
    def data_weights(self) -> np.ndarray:
        return np.array([d.data_weight for d in self.data])

    @property
    def targets(self) -> np.ndarray:
        return np.array([d.targets for d in self.data])

    @targets.setter
    def targets(self, targets: np.ndarray) -> None:
        """Set the targets for each molecule in the dataset

        Parameters
        ----------
        targets : np.ndarray
            the targets for each molecule. NOTE: must be the same length as the underlying dataset.
        """
        if not len(self.data) == len(targets):
            raise ValueError(
                "number of molecules and targets must be of same length! "
                f"num molecules: {len(self.data)}, num targets: {len(targets)}"
            )

        for i in range(len(self.data)):
            self.data[i].targets = targets[i]

    @property
    def gt_targets(self) -> np.ndarray:
        if len(self.data) > 0 and self.data[0].gt_targets is None:
            return None

        return np.array([d.gt_targets for d in self.data])

    @property
    def lt_targets(self) -> List[np.ndarray]:
        if len(self.data) > 0 and self.data[0].lt_targets is None:
            return None 
        
        return np.array([d.lt_targets for d in self.data])

    @property
    def num_tasks(self) -> Optional[int]:
        return self.data[0].num_tasks() if len(self.data) > 0 else None

    @property
    def features_size(self) -> Optional[int]:
        if len(self.data) > 0 and self.data[0].features is None:
            return None
        
        return len(self.data[0].features)

    @property
    def atom_descriptors_size(self) -> int:
        if len(self.data) > 0 and self.data[0].atom_descriptors is None:
            return None
        
        return len(self.data[0].atom_descriptors[0])

    @property
    def atom_features_size(self) -> int:
        if len(self.data) > 0 and self.data[0].atom_features is None:
            return None

        return len(self.data[0].atom_features[0])

    @property
    def bond_features_size(self) -> int:
        """
        Returns the size of custom additional bond features vector associated with the molecules.

        :return: The size of the additional bond feature vector.
        """
        if len(self.data) > 0 and self.data[0].bond_features is None:
            return None
        
        return len(self.data[0].bond_features[0])

    def normalize_features(
        self,
        scaler: StandardScaler = None,
        replace_nan_token: int = 0,
        scale_atom_descriptors: bool = False,
        scale_bond_features: bool = False,
    ) -> StandardScaler:
        """
        Normalizes the features of the dataset using a :class:`~chemprop.data.StandardScaler`.

        The :class:`~chemprop.data.StandardScaler` subtracts the mean and divides by the standard 
        deviation for each feature independently.

        If a :class:`~chemprop.data.StandardScaler` is provided, it is used to perform the 
        normalization. Otherwise, a :class:`~chemprop.data.StandardScaler` is first fit to the 
        features in this  dataset and is then used to perform the normalization.

        :param scaler: A fitted :class:`~chemprop.data.StandardScaler`. If it is provided it is 
            used, otherwise a new :class:`~chemprop.data.StandardScaler` is first fitted to this 
            data and is then used.
        :param replace_nan_token: A token to use to replace NaN entries in the features.
        :param scale_atom_descriptors: If the features that need to be scaled are atom features 
            rather than molecule.
        :param scale_bond_features: If the features that need to be scaled are bond descriptors 
            rather than molecule.
        :return: A fitted :class:`~chemprop.data.StandardScaler`. If a :class:`~chemprop.data.
            StandardScaler` is provided as a parameter, this is the same :class:`~chemprop.data.
            StandardScaler`. Otherwise, this is a new :class:`~chemprop.data.StandardScaler` that 
            has been fit on this dataset.
        """
        if len(self.data) == 0 or (
            self.data[0].features is None
            and not scale_bond_features
            and not scale_atom_descriptors
        ):
            return None

        if scaler is None:
            if scale_atom_descriptors and not self.data[0].atom_descriptors is None:
                features = np.vstack([d.raw_atom_descriptors for d in self.data])
            elif scale_atom_descriptors and not self.data[0].atom_features is None:
                features = np.vstack([d.raw_atom_features for d in self.data])
            elif scale_bond_features:
                features = np.vstack([d.raw_bond_features for d in self.data])
            else:
                features = np.vstack([d.raw_features for d in self.data])
            scaler = StandardScaler(replace_nan_token=replace_nan_token)
            scaler.fit(features)

        if scale_atom_descriptors and not self.data[0].atom_descriptors is None:
            for d in self.data:
                d.atom_descriptors = scaler.transform(d.raw_atom_descriptors)
        elif scale_atom_descriptors and not self.data[0].atom_features is None:
            for d in self.data:
                d.atom_features = scaler.transform(d.raw_atom_features)
        elif scale_bond_features:
            for d in self.data:
                d.bond_features = scaler.transform(d.raw_bond_features)
        else:
            for d in self.data:
                d.features = scaler.transform(d.raw_features.reshape(1, -1))[0]

        return scaler

    def normalize_targets(self) -> StandardScaler:
        """
        Normalizes the targets of the dataset using a :class:`~chemprop.data.StandardScaler`.

        The :class:`~chemprop.data.StandardScaler` subtracts the mean and divides by the standard deviation
        for each task independently.

        This should only be used for regression datasets.

        :return: A :class:`~chemprop.data.StandardScaler` fitted to the targets.
        """
        targets = [d.raw_targets for d in self.data]
        scaler = StandardScaler().fit(targets)
        scaled_targets = scaler.transform(targets)
        self.targets = scaled_targets

        return scaler

    def reset_features_and_targets(self) -> None:
        """Resets the features (atom, bond, and molecule) and targets to their raw values."""
        for d in self.data:
            d.reset_features_and_targets()
