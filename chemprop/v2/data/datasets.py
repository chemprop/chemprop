from dataclasses import dataclass, field
from functools import cached_property
from typing import NamedTuple

import numpy as np
from rdkit import Chem
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset

from chemprop.v2.data.datapoints import MoleculeDatapoint, ReactionDatapoint
from chemprop.v2.featurizers import (
    MolGraph,
    MoleculeMolGraphFeaturizerProto,
    MoleculeMolGraphFeaturizer,
    RxnMolGraphFeaturizerProto,
    RxnMolGraphFeaturizer,
)


class Datum(NamedTuple):
    """a singular training data point"""
    mg: MolGraph
    V_d: np.ndarray | None
    x_f: np.ndarray | None
    y: np.ndarray | None
    weight: float
    lt_mask: np.ndarray | None
    gt_mask: np.ndarray | None


class _MolGraphDatasetMixin:
    def __len__(self) -> int:
        return len(self.data)

    @cached_property
    def _Y(self) -> np.ndarray:
        """the raw targets of the dataset"""
        return np.array([d.y for d in self.data])

    @property
    def Y(self) -> np.ndarray:
        """the (scaled) targets of the dataset"""
        return self.__Y

    @Y.setter
    def Y(self, Y: np.ndarray):
        self._validate_attribute(Y, "targets")

        self.__Y = Y

    @cached_property
    def _X_f(self) -> np.ndarray:
        """the raw molecule features of the dataset"""
        return np.array([d.x_f for d in self.data])

    @property
    def X_f(self) -> np.ndarray:
        """the (scaled) molecule features of the dataset"""
        return self.__X_f

    @X_f.setter
    def X_f(self, X_f: np.ndarray):
        self._validate_attribute(X_f, "molecule features")

        self.__X_f = X_f

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

    def normalize_targets(self, scaler: StandardScaler | None = None) -> StandardScaler:
        """Normalizes the targets of the dataset using a :obj:`StandardScaler`

        The :obj:`StandardScaler` subtracts the mean and divides by the standard deviation for
        each task independently. NOTE: This should only be used for regression datasets.

        Returns
        -------
        StandardScaler
            a scaler fit to the targets.
        """
        scaler = scaler or StandardScaler()
        self.Y = scaler.fit_transform(self._Y)

        return scaler

    def normalize_inputs(
        self, key: str | None = "X_f", scaler: StandardScaler | None = None
    ) -> StandardScaler:
        VALID_KEYS = {"X_f", None}
        if key not in VALID_KEYS:
            raise ValueError(f"Invalid feature key! got: {key}. expected one of: {VALID_KEYS}")

        X = self.X_f

        if scaler is None:
            scaler = StandardScaler().fit(X)

        return scaler

    def reset(self):
        """Reset the {atom, bond, molecule} features and targets of each datapoint to its
        initial, unnormalized values."""
        self.__Y = self._Y
        self.__X_f = self._X_f

    def _validate_attribute(self, X: np.ndarray, label: str):
        if not len(self.data) == len(X):
            raise ValueError(
                f"number of molecules ({len(self.data)}) and {label} ({len(X)}) "
                "must have same length!"
            )


@dataclass
class MoleculeDataset(Dataset, _MolGraphDatasetMixin):
    """A `MolgraphDataset` composed of `MoleculeDatapoint`s

    Parameters
    ----------
    data : Iterable[MoleculeDatapoint]
        the data from which to create a dataset
    featurizer : MoleculeFeaturizer
        the featurizer with which to generate MolGraphs of the molecules
    """

    data: list[MoleculeDatapoint]
    featurizer: MoleculeMolGraphFeaturizerProto = field(default_factory=MoleculeMolGraphFeaturizer)

    def __post_init__(self):
        self.reset()

    def __getitem__(self, idx: int) -> Datum:
        d = self.data[idx]
        mg = self.featurizer(d.mol, self.V_fs[idx], self.E_fs[idx])

        return Datum(mg, self.V_ds[idx], self.X_f[idx], self.Y[idx], d.weight, d.lt_mask, d.gt_mask)

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
        return np.array([d.V_f for d in self.data])

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
        return np.array([d.E_f for d in self.data])

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
        return np.array([d.V_d for d in self.data])

    @property
    def V_ds(self) -> list[np.ndarray]:
        """the (scaled) atom descriptors of the dataset"""
        return self.__V_ds

    @V_ds.setter
    def V_ds(self, V_ds: list[np.ndarray] | None):
        self._validate_attribute(V_ds, "atom descriptors")

        self.__V_ds = V_ds

    @property
    def d_vf(self) -> int | None:
        """the extra atom feature dimension, if any"""
        return None if self.V_fs[0] is None else self.V_fs[0].shape[1]

    @property
    def d_ef(self) -> int | None:
        """the extra bond feature dimension, if any"""
        return None if self.E_fs is None else self.E_fs[0].shape[1]

    @property
    def d_vd(self) -> int | None:
        """the extra atom descriptor dimension, if any"""
        return None if self.V_ds is None else self.V_ds[0].shape[1]

    def normalize_inputs(
        self, key: str | None = "X_f", scaler: StandardScaler | None = None
    ) -> StandardScaler:
        VALID_KEYS = {"X_f", "V_f", "E_f", "V_d", None}

        match key:
            case "X_f":
                X = self.X_f
            case "V_f":
                X = None if self.V_fs is None else np.concatenate(self.V_fs, axis=0)
            case "E_f":
                X = None if self.E_fs is None else np.concatenate(self.E_fs, axis=0)
            case "V_d":
                X = None if self.V_ds is None else np.concatenate(self.V_ds, axis=0)
            case None:
                return [self.normalize_inputs(k, scaler) for k in VALID_KEYS - {None}]
            case _:
                ValueError(f"Invalid feature key! got: {key}. expected one of: {VALID_KEYS}")

        if X is None:
            return scaler

        if scaler is None:
            scaler = StandardScaler().fit(X)

        match key:
            case "X_f":
                self.X_f = scaler.transform(X)
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
        """reset the {atom, bond, molecule} features and targets of each datapoint to its raw
        value"""
        super().reset()
        self.__V_fs = self._V_fs
        self.__E_fs = self._E_fs
        self.__V_ds = self._V_ds


@dataclass
class ReactionDataset(Dataset, _MolGraphDatasetMixin):
    """A :class:`ReactionDataset` composed of :class:`ReactionDatapoint`s"""

    data: list[ReactionDatapoint]
    """the dataset from which to load"""
    featurizer: RxnMolGraphFeaturizerProto = field(default_factory=RxnMolGraphFeaturizer)
    """the featurizer with which to generate MolGraphs of the input"""

    def __getitem__(self, idx: int) -> Datum:
        d = self.data[idx]
        mg = self.featurizer(((d.rct, d.pdt)), None, None)

        return Datum(mg, None, d.x_f, d.y, d.weight, d.lt_mask, d.gt_mask)

    @property
    def smiles(self) -> list[str]:
        return [(Chem.MolToSmiles(d.rct), Chem.MolToSmiles(d.pdt)) for d in self.data]

    @property
    def mols(self) -> list[Chem.Mol]:
        return [(d.rct, d.pdt) for d in self.data]
