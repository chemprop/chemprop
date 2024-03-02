from abc import abstractmethod
from dataclasses import dataclass, field
from functools import cached_property
from typing import NamedTuple

import numpy as np
from numpy.typing import ArrayLike
from rdkit import Chem
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset

from chemprop.featurizers import (
    MolGraph,
    MoleculeMolGraphFeaturizer,
    SimpleMoleculeMolGraphFeaturizer,
    RxnMolGraphFeaturizer,
    CGRFeaturizer,
)
from chemprop.data.datapoints import MoleculeDatapoint, ReactionDatapoint


class Datum(NamedTuple):
    """a singular training data point"""

    mg: MolGraph
    V_d: np.ndarray | None
    x_d: np.ndarray | None
    y: np.ndarray | None
    weight: float
    lt_mask: np.ndarray | None
    gt_mask: np.ndarray | None


class MolGraphDataset(Dataset):
    @abstractmethod
    def __getitem__(self, idx) -> Datum:
        pass


class _MolGraphDatasetMixin:
    def __len__(self) -> int:
        return len(self.data)

    @cached_property
    def _Y(self) -> np.ndarray:
        """the raw targets of the dataset"""
        return np.array([d.y for d in self.data], float)

    @property
    def Y(self) -> np.ndarray:
        """the (scaled) targets of the dataset"""
        return self.__Y

    @Y.setter
    def Y(self, Y: ArrayLike):
        self._validate_attribute(Y, "targets")

        self.__Y = np.array(Y, float)

    @cached_property
    def _X_d(self) -> np.ndarray:
        """the raw extra descriptors of the dataset"""
        return np.array([d.x_d for d in self.data])

    @property
    def X_d(self) -> np.ndarray:
        """the (scaled) extra descriptors of the dataset"""
        return self.__X_d

    @X_d.setter
    def X_d(self, X_d: ArrayLike):
        self._validate_attribute(X_d, "extra descriptors")

        self.__X_d = np.array(X_d)

    @property
    def weights(self) -> np.ndarray:
        return np.array([d.weight for d in self.data])

    @property
    def gt_mask(self) -> np.ndarray:
        return np.array([d.gt_mask for d in self.data])

    @property
    def lt_mask(self) -> np.ndarray:
        return np.array([d.lt_mask for d in self.data])

    @property
    def t(self) -> int | None:
        return self.data[0].t if len(self.data) > 0 else None

    @property
    def d_xd(self) -> int:
        """the extra molecule descriptor dimension, if any"""
        return 0 if np.equal(self.X_d, None).all() else self.X_d.shape[1]

    def normalize_targets(self, scaler: StandardScaler | None = None) -> StandardScaler:
        """Normalizes the targets of this dataset using a :obj:`StandardScaler`

        The :obj:`StandardScaler` subtracts the mean and divides by the standard deviation for
        each task independently. NOTE: This should only be used for regression datasets.

        Returns
        -------
        StandardScaler
            a scaler fit to the targets.
        """
        if scaler is None:
            scaler = StandardScaler()
            self.Y = scaler.fit_transform(self._Y)
        else:
            self.Y = scaler.transform(self._Y)

        return scaler

    def normalize_inputs(
        self, key: str | None = "X_d", scaler: StandardScaler | None = None
    ) -> StandardScaler:
        VALID_KEYS = {"X_d", None}
        if key not in VALID_KEYS:
            raise ValueError(f"Invalid feature key! got: {key}. expected one of: {VALID_KEYS}")

        X = self.X_d

        if scaler is None:
            scaler = StandardScaler()
            self.X_d = scaler.fit_transform(X)

        return scaler

    def reset(self):
        """Reset the atom and bond features; atom and extra descriptors; and targets of each
        datapoint to their initial, unnormalized values."""
        self.__Y = self._Y
        self.__X_d = self._X_d

    def _validate_attribute(self, X: np.ndarray, label: str):
        if not len(self.data) == len(X):
            raise ValueError(
                f"number of molecules ({len(self.data)}) and {label} ({len(X)}) "
                "must have same length!"
            )


@dataclass
class MoleculeDataset(_MolGraphDatasetMixin, MolGraphDataset):
    """A :class:`MolgraphDataset` composed of :class:`MoleculeDatapoint`\s

    Parameters
    ----------
    data : Iterable[MoleculeDatapoint]
        the data from which to create a dataset
    featurizer : MoleculeFeaturizer
        the featurizer with which to generate MolGraphs of the molecules
    """

    data: list[MoleculeDatapoint]
    featurizer: MoleculeMolGraphFeaturizer = field(default_factory=SimpleMoleculeMolGraphFeaturizer)

    def __post_init__(self):
        if self.data is None:
            raise ValueError("Data cannot be None!")

        self.reset()

    def __getitem__(self, idx: int) -> Datum:
        d = self.data[idx]
        mg = self.featurizer(d.mol, self.V_fs[idx], self.E_fs[idx])

        return Datum(mg, self.V_ds[idx], self.X_d[idx], self.Y[idx], d.weight, d.lt_mask, d.gt_mask)

    @property
    def smiles(self) -> list[str]:
        """the SMILES strings associated with the dataset"""
        return [Chem.MolToSmiles(d.mol) for d in self.data]

    @property
    def mols(self) -> list[Chem.Mol]:
        """the molecules associated with the dataset"""
        return [d.mol for d in self.data]

    @property
    def _V_fs(self) -> list[np.ndarray]:
        """the raw atom features of the dataset"""
        return [d.V_f for d in self.data]

    @property
    def V_fs(self) -> list[np.ndarray]:
        """the (scaled) atom descriptors of the dataset"""
        return self.__V_fs

    @V_fs.setter
    def V_fs(self, V_fs: list[np.ndarray]):
        """the (scaled) atom features of the dataset"""
        self._validate_attribute(V_fs, "atom features")

        self.__V_fs = V_fs

    @property
    def _E_fs(self) -> list[np.ndarray]:
        """the raw bond features of the dataset"""
        return [d.E_f for d in self.data]

    @property
    def E_fs(self) -> list[np.ndarray]:
        """the (scaled) bond features of the dataset"""
        return self.__E_fs

    @E_fs.setter
    def E_fs(self, E_fs: list[np.ndarray]):
        self._validate_attribute(E_fs, "bond features")

        self.__E_fs = E_fs

    @property
    def _V_ds(self) -> list[np.ndarray]:
        """the raw atom descriptors of the dataset"""
        return [d.V_d for d in self.data]

    @property
    def V_ds(self) -> list[np.ndarray]:
        """the (scaled) atom descriptors of the dataset"""
        return self.__V_ds

    @V_ds.setter
    def V_ds(self, V_ds: list[np.ndarray]):
        self._validate_attribute(V_ds, "atom descriptors")

        self.__V_ds = V_ds

    @property
    def d_vf(self) -> int:
        """the extra atom feature dimension, if any"""
        return 0 if all(np.equal(V_f, None).all() for V_f in self.V_fs) else self.V_fs[0].shape[1]

    @property
    def d_ef(self) -> int:
        """the extra bond feature dimension, if any"""
        return 0 if all(np.equal(E_f, None).all() for E_f in self.E_fs) else self.E_fs[0].shape[1]

    @property
    def d_vd(self) -> int:
        """the extra atom descriptor dimension, if any"""
        return 0 if all(np.equal(V_d, None).all() for V_d in self.V_ds) else self.V_ds[0].shape[1]

    def normalize_inputs(
        self, key: str | None = "X_d", scaler: StandardScaler | None = None
    ) -> StandardScaler:
        VALID_KEYS = {"X_d", "V_f", "E_f", "V_d", None}

        match key:
            case "X_d":
                X = None if np.all(self.X_d == None) else self.X_d
            case "V_f":
                X = None if np.all(self.V_fs == None) else np.concatenate(self.V_fs, axis=0)
            case "E_f":
                X = None if np.all(self.E_fs == None) else np.concatenate(self.E_fs, axis=0)
            case "V_d":
                X = None if np.all(self.V_ds == None) else np.concatenate(self.V_ds, axis=0)
            case None:
                return [self.normalize_inputs(k, scaler) for k in VALID_KEYS - {None}]
            case _:
                raise ValueError(f"Invalid feature key! got: {key}. expected one of: {VALID_KEYS}")

        if X is None:
            return scaler

        if scaler is None:
            scaler = StandardScaler().fit(X)

        match key:
            case "X_d":
                self.X_d = scaler.transform(X)
            case "V_f":
                self.V_fs = [scaler.transform(V_f) for V_f in self.V_fs]
            case "E_f":
                self.E_fs = [scaler.transform(E_f) for E_f in self.E_fs]
            case "V_d":
                self.V_ds = [scaler.transform(V_d) for V_d in self.V_ds]
            case _:
                raise RuntimeError("unreachable code reached!")

        return scaler

    def reset(self):
        """Reset the atom and bond features; atom and extra descriptors; and targets of each
        datapoint to their initial, unnormalized values."""
        super().reset()
        self.__V_fs = self._V_fs
        self.__E_fs = self._E_fs
        self.__V_ds = self._V_ds


@dataclass
class ReactionDataset(_MolGraphDatasetMixin, MolGraphDataset):
    """A :class:`ReactionDataset` composed of :class:`ReactionDatapoint`\s"""

    data: list[ReactionDatapoint]
    """the dataset from which to load"""
    featurizer: RxnMolGraphFeaturizer = field(default_factory=CGRFeaturizer)
    """the featurizer with which to generate MolGraphs of the input"""

    def __post_init__(self):
        if self.data is None:
            raise ValueError("Data cannot be None!")

        self.reset()

    def __getitem__(self, idx: int) -> Datum:
        d = self.data[idx]
        mg = self.featurizer((d.rct, d.pdt), None, None)

        return Datum(mg, None, d.x_d, d.y, d.weight, d.lt_mask, d.gt_mask)

    @property
    def smiles(self) -> list[str]:
        return [(Chem.MolToSmiles(d.rct), Chem.MolToSmiles(d.pdt)) for d in self.data]

    @property
    def mols(self) -> list[Chem.Mol]:
        return [(d.rct, d.pdt) for d in self.data]


@dataclass(repr=False, eq=False)
class MulticomponentDataset(_MolGraphDatasetMixin, Dataset):
    """A :class:`MulticomponentDataset` is a :class:`Dataset` composed of parallel :class:`MoleculeDatasets` and :class:`ReactionDataset`\s"""

    datasets: list[MoleculeDataset | ReactionDataset]
    """the parallel datasets"""

    def __post_init__(self):
        sizes = [len(dset) for dset in self.datasets]
        if not all(sizes[0] == size for size in sizes[1:]):
            raise ValueError(f"Datasets must have all same length! got: {sizes}")

    def __len__(self) -> int:
        return len(self.datasets[0])

    @property
    def n_components(self) -> int:
        return len(self.datasets)

    def __getitem__(self, idx: int) -> list[Datum]:
        return [dset[idx] for dset in self.datasets]

    @property
    def smiles(self) -> list[list[str]]:
        return list(zip(*[dset.smiles for dset in self.datasets]))

    @property
    def mols(self) -> list[list[Chem.Mol]]:
        return list(zip(*[dset.mols for dset in self.datasets]))

    def normalize_targets(self, scaler: StandardScaler | None = None) -> StandardScaler:
        return self.datasets[0].normalize_targets(scaler)

    def normalize_inputs(
        self, key: str | None = "X_d", scaler: StandardScaler | None = None
    ) -> list[StandardScaler]:
        return [dset.normalize_inputs(key, scaler) for dset in self.datasets]

    def reset(self):
        return [dset.reset() for dset in self.datasets]
