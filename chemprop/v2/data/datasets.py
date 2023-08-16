from abc import abstractmethod
from typing import Iterable

import numpy as np
from rdkit import Chem
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset

from chemprop.v2.data.datapoints import DatapointBase, MoleculeDatapoint, ReactionDatapoint
from chemprop.v2.featurizers import (
    MolGraph,
    MoleculeFeaturizerProto,
    MoleculeFeaturizer,
    ReactionFeaturizerProto,
    ReactionFeaturizer,
)

Datum = tuple[MolGraph, np.ndarray, np.ndarray, np.ndarray, float, np.ndarray, np.ndarray]


class MolGraphDatasetMixin:
    def __len__(self) -> int:
        return len(self.data)

    @property
    def _Y(self) -> np.ndarray:
        return np.array([d._y for d in self.data])

    @property
    def Y(self) -> np.ndarray:
        return np.array([d.y for d in self.data])

    @Y.setter
    def Y(self, Y: np.ndarray) -> None:
        if not len(self.data) == len(Y):
            raise ValueError(
                "number of molecules and targets must be of same length! "
                f"num molecules: {len(self.data)}, num targets: {len(Y)}"
            )

        for d, y in zip(self.data, Y):
            d.y = y

    @property
    def X_v(self) -> np.ndarray | None:
        if len(self.data) > 0 and self.data[0].x_v is None:
            return None

        return np.array([d.x_v for d in self.data])

    @X_v.setter
    def X_v(self, X: np.ndarray):
        if not len(self.data) == len(X):
            raise ValueError(
                "number of molecules and features must be of same length! "
                f"num molecules: {len(self.data)}, num features: {len(X)}"
            )

        for d, x in zip(self.data, X):
            d.x_v = x

    @property
    def weights(self) -> np.ndarray:
        return np.array([d.weight for d in self.data])

    @property
    def gt_mask(self) -> np.ndarray | None:
        if len(self.data) > 0 and self.data[0].gt_mask is None:
            return None

        return np.array([d.gt_mask for d in self.data])

    @property
    def lt_mask(self) -> np.ndarray | None:
        if len(self.data) > 0 and self.data[0].lt_mask is None:
            return None

        return np.array([d.lt_mask for d in self.data])

    @property
    def t(self) -> int | None:
        return self.data[0].t if len(self.data) > 0 else None

    @property
    def d_v(self) -> int | None:
        if len(self.data) > 0 and self.data[0].x_v is None:
            return None

        return len(self.data[0].x_v)

    def normalize_targets(self, scaler: StandardScaler | None = None) -> StandardScaler:
        """Normalizes the targets of the dataset using a `StandardScaler`

        The StandardScaler subtracts the mean and divides by the standard deviation for each task
        independently. NOTE: This should only be used for regression datasets.

        Returns
        -------
        StandardScaler
            a scaler fit to the targets.
        """
        Y = np.array([d._y for d in self.data])

        scaler = (scaler or StandardScaler()).fit(Y)
        self.Y = scaler.transform(Y)

        return scaler

    def normalize_inputs(
        self, key: str | None = "X_v", scaler: StandardScaler | None = None
    ) -> StandardScaler:
        VALID_KEYS = {"X_v", None}
        if key not in VALID_KEYS:
            raise ValueError(f"Invalid feature key! got: {key}. expected one of: {VALID_KEYS}")

        X = self.X_v

        if scaler is None:
            scaler = StandardScaler().fit(X)

        return scaler

    def reset(self) -> None:
        """Reset the {atom, bond, molecule} features and targets of each datapoint to its
        initial, unnormalized values."""
        [d.reset() for d in self.data]


class MoleculeDataset(MolGraphDatasetMixin):
    """A `MolgraphDataset` composed of `MoleculeDatapoint`s

    Parameters
    ----------
    data : Iterable[MoleculeDatapoint]
        the data from which to create a dataset
    featurizer : MoleculeFeaturizer
        the featurizer with which to generate MolGraphs of the molecules
    """

    def __init__(
        self, data: Iterable[MoleculeDatapoint], featurizer: MoleculeFeaturizerProto | None
    ):
        self.data = list(data)
        self.featurizer = featurizer or MoleculeFeaturizer()

    def __getitem__(self, idx: int) -> Datum:
        d = self.data[idx]

        return (
            self.featurizer(d.mol, d.V_f, d.E_f),
            d.V_d,
            d.x_v,
            d.y,
            d.weight,
            d.lt_mask,
            d.gt_mask,
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
    def V_fs(self) -> list[np.ndarray] | None:
        if len(self.data) > 0 and self.data[0].V_f is None:
            return None

        return [d.V_f for d in self.data]

    @V_fs.setter
    def V_fs(self, V_fs: list[np.ndarray]):
        if not len(self.data) == len(V_fs):
            raise ValueError(
                "number of molecules and supplied atom features must be of same length! "
                f"expected: {len(self.data)}. got: {len(V_fs)}"
            )
        for d, V_f in zip(self.data, V_fs):
            d.V_f = V_f

    @property
    def E_fs(self) -> list[np.ndarray] | None:
        if len(self.data) > 0 and self.data[0].E_f is None:
            return None

        return [d.E_f for d in self.data]

    @E_fs.setter
    def E_fs(self, E_fs: list[np.ndarray]):
        if not len(self.data) == len(E_fs):
            raise ValueError(
                "number of molecules and supplied edge features must be of same length! "
                f"expected: {len(self.data)}. got: {len(E_fs)}"
            )
        for d, E_f in zip(self.data, E_fs):
            d.V_d = E_f

    @property
    def V_ds(self) -> list[np.ndarray] | None:
        if len(self.data) > 0 and self.data[0].V_d is None:
            return None

        return [d.V_d for d in self.data]

    @V_ds.setter
    def V_ds(self, V_ds: list[np.ndarray]):
        if not len(self.data) == len(V_ds):
            raise ValueError(
                "number of molecules and supplied atom descriptors must be of same length! "
                f"expected: {len(self.data)}. got: {len(V_ds)}"
            )
        for d, V_d in zip(self.data, V_ds):
            d.V_d = V_d

    @property
    def d_vf(self) -> int | None:
        if len(self.data) > 0 and self.data[0].V_f is None:
            return None

        return len(self.data[0].V_f[0])

    @property
    def d_ef(self) -> int | None:
        if len(self.data) > 0 and self.data[0].E_f is None:
            return None

        return len(self.data[0].E_f[0])

    @property
    def d_vd(self) -> int | None:
        if len(self.data) > 0 and self.data[0].V_d is None:
            return None

        return len(self.data[0].V_d[0])

    def normalize_inputs(
        self, key: str | None = "X_v", scaler: StandardScaler | None = None
    ) -> StandardScaler:
        VALID_KEYS = {"X_v", "V_f", "E_f", "V_d", None}
        if key not in VALID_KEYS:
            raise ValueError(f"Invalid feature key! got: {key}. expected one of: {VALID_KEYS}")

        if key == "X_v":
            X = self.X_v
        elif key == "V_f":
            X = None if self.V_fs is None else np.concatenate(self.V_fs, axis=0)
        elif key == "E_f":
            X = None if self.E_fs is None else np.concatenate(self.E_fs, axis=0)
        elif key == "V_d":
            X = None if self.V_ds is None else np.concatenate(self.V_ds, axis=0)
        elif key is None:
            return [self.normalize_inputs(k, scaler) for k in VALID_KEYS - {None}]

        if X is None:
            return scaler

        if scaler is None:
            scaler = StandardScaler().fit(X)

        if key == "X_v":
            self.X_v = scaler.transform(X)
        elif key == "V_f":
            self.V_fs = [scaler.transform(V_f) for V_f in self.V_fs]
        elif key == "E_f":
            self.E_fs = [scaler.transform(E_f) for E_f in self.E_fs]
        elif key == "V_d":
            self.V_ds = [scaler.transform(V_d) for V_d in self.V_ds]

        return scaler


class ReactionDataset(MolGraphDatasetMixin):
    """A `MolgraphDataset` composed of `ReactionDatapoint`s

    Parameters
    ----------
    data : Iterable[ReactionDatapoint]
        the dataset from which to load
    featurizer : ReactionFeaturizer
        the featurizer with which to generate MolGraphs of the input
    """

    def __init__(
        self, data: Iterable[ReactionDatapoint], featurizer: ReactionFeaturizerProto | None
    ):
        self.data = list(data)
        self.featurizer = featurizer or ReactionFeaturizer()

    def __getitem__(self, idx: int) -> Datum:
        d = self.data[idx]

        return (
            self.featurizer(((d.rct_mol, d.pdt_mol)), None, None),
            None,
            d.x_v,
            d.y,
            d.weight,
            d.lt_mask,
            d.gt_mask,
        )

    @property
    def smiles(self) -> list[str]:
        return [(d.rct_smi, d.pdt_smi) for d in self.data]

    @property
    def mols(self) -> list[Chem.Mol]:
        return [(d.rct_mol, d.pdt_mol) for d in self.data]

    @property
    def n_mols(self) -> int:
        return len(self.data[0])
