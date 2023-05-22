from __future__ import annotations

from abc import abstractmethod
from typing import Iterable, Optional

import numpy as np
from rdkit import Chem
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset

from chemprop.v2.data.datapoints import DatapointBase, MoleculeDatapoint, ReactionDatapoint
from chemprop.v2.featurizers import MolGraph, MoleculeFeaturizerBase, MoleculeFeaturizer, ReactionFeaturizer, ReactionFeaturizerBase

Datum = tuple[MolGraph, np.ndarray, np.ndarray, np.ndarray, float, np.ndarray, np.ndarray]


class MolGraphDatasetBase(Dataset):
    def __init__(self, data: Iterable[DatapointBase]):
        super().__init__()

        self.data = list(data)

    @abstractmethod
    def __getitem__(self, idx: int) -> Datum:
        pass

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

    # @property
    # def X_phase(self) -> Optional[np.ndarray]:
    #     if len(self.data) > 0 and self.data[0].x_phase is None:
    #         return None

    #     return np.ndarray([d.x_phase for d in self.data])

    @property
    def weights(self) -> np.ndarray:
        return np.array([d.weight for d in self.data])

    @property
    def gt_mask(self) -> Optional[np.ndarray]:
        if len(self.data) > 0 and self.data[0].gt_mask is None:
            return None

        return np.array([d.gt_mask for d in self.data])

    @property
    def lt_mask(self) -> Optional[np.ndarray]:
        if len(self.data) > 0 and self.data[0].lt_mask is None:
            return None

        return np.array([d.lt_mask for d in self.data])

    @property
    def t(self) -> Optional[int]:
        return self.data[0].t if len(self.data) > 0 else None

    @property
    def d_v(self) -> Optional[int]:
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
        targets = np.array([d._y for d in self.data])
        scaler = (scaler or StandardScaler()).fit(targets)
        scaled_targets = scaler.transform(targets)
        self.Y = scaled_targets

        return scaler

    def reset(self) -> None:
        """Resets the features (atom, bond, and molecule) and targets of each datapoint to its 
        initial, unnormalized values."""
        [d.reset() for d in self.data]


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
    def V_f(self) -> list[np.ndarray] | None:
        if len(self.data) > 0 and self.data[0].V_f is None:
            return None

        return [d.V_f for d in self.data]

    @V_f.setter
    def V_f(self, X: np.ndarray):
        if not len(self.data) == len(X):
            raise ValueError(
                "number of molecules and atom features must be of same length! "
                f"expected: `{len(self.data)} x *`. got: `{len(X)} x *`"
            )
        for d, x in zip(self.data, X):
            d.V_f = x

    @property
    def E_f(self) -> list[np.ndarray] | None:
        if len(self.data) > 0 and self.data[0].E_f is None:
            return None

        return [d.E_f for d in self.data]

    @E_f.setter
    def E_f(self, X: np.ndarray):
        if not len(self.data) == len(X):
            raise ValueError(
                "number of molecules and edge features must be of same length! "
                f"expected: `{len(self.data)} x *`. got: `{len(X)} x *`"
            )
        for d, E_f in zip(self.data, X):
            d.V_d = E_f

    @property
    def V_d(self) -> list[np.ndarray] | None:
        if len(self.data) > 0 and self.data[0].V_d is None:
            return None

        return [d.V_d for d in self.data]

    @V_d.setter
    def V_d(self, X: np.ndarray):
        if not len(self.data) == len(X):
            raise ValueError(
                "number of molecules and atom descriptors must be of same length! "
                f"expected: `{len(self.data)} x *`. got: `{len(X)} x *`"
            )
        for d, V_d in zip(self.data, X):
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

    def normalize(
        self, key: str | None = "X_v", scaler: StandardScaler | None = None
    ) -> StandardScaler:
        VALID_KEYS = {"X_v", "V_f", "E_f", "V_d", "all", None}
        if key not in VALID_KEYS:
            raise ValueError(f"Invalid feature key! got: {key}. expected one of: {VALID_KEYS}")

        if key == "X_v":
            X = self.X_v
        elif key == "V_f":
            X = self.V_f
        elif key == "E_f":
            X = self.E_f
        elif key == "V_d":
            X = self.V_d
        elif key is None or key == "all":
            return [self.normalize(k, scaler) for k in VALID_KEYS - {None, "all"}]

        if X is None:
            return scaler

        if scaler is None:
            scaler = StandardScaler().fit(X)
        X_ = scaler.transform(X)

        if key == "X_v":
            self.X_v = X_
        elif key == "V_f":
            self.V_f = X_
        elif key == "E_f":
            self.E_f = X_
        elif key == "V_d":
            self.E_f = X_

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
        self, data: Iterable[ReactionDatapoint], featurizer: ReactionFeaturizerBase | None
    ):
        super().__init__(data)
        self.featurizer = featurizer or ReactionFeaturizer()

    def __getitem__(self, idx: int) -> Datum:
        d = self.data[idx]

        return (
            self.featurizer(d.mols, d.V_f, d.E_f),
            None,
            d.x_v,
            d.y,
            d.weight,
            d.lt_mask,
            d.gt_mask,
        )

    @property
    def smiles(self) -> list[str]:
        return [d.smis for d in self.data]

    @property
    def mols(self) -> list[Chem.Mol]:
        return [d.mols for d in self.data]

    @property
    def n_mol(self) -> int:
        return len(self.data[0])
