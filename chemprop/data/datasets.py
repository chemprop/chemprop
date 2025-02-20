from dataclasses import dataclass, field
from functools import cached_property
from typing import NamedTuple, TypeAlias

import numpy as np
from numpy.typing import ArrayLike
from rdkit import Chem
from rdkit.Chem import Mol
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset

from chemprop.data.datapoints import MoleculeDatapoint, ReactionDatapoint
from chemprop.data.molgraph import MolGraph
from chemprop.featurizers.base import Featurizer
from chemprop.featurizers.molgraph import CGRFeaturizer, SimpleMoleculeMolGraphFeaturizer
from chemprop.featurizers.molgraph.cache import MolGraphCache, MolGraphCacheOnTheFly
from chemprop.types import Rxn


class Datum(NamedTuple):
    """a singular training data point"""

    mg: MolGraph
    V_d: np.ndarray | None
    E_d: np.ndarray | None
    x_d: np.ndarray | None
    y: np.ndarray | None
    weight: float
    lt_mask: np.ndarray | None
    gt_mask: np.ndarray | None


MolGraphDataset: TypeAlias = Dataset[Datum]


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
        return 0 if self.X_d[0] is None else self.X_d.shape[1]

    @property
    def names(self) -> list[str]:
        return [d.name for d in self.data]

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
            scaler = StandardScaler().fit(self._Y)

        self.Y = scaler.transform(self._Y)

        return scaler

    def normalize_inputs(
        self, key: str = "X_d", scaler: StandardScaler | None = None
    ) -> StandardScaler:
        VALID_KEYS = {"X_d"}
        if key not in VALID_KEYS:
            raise ValueError(f"Invalid feature key! got: {key}. expected one of: {VALID_KEYS}")

        X = self.X_d if self.X_d[0] is not None else None

        if X is None:
            return scaler

        if scaler is None:
            scaler = StandardScaler().fit(X)

        self.X_d = scaler.transform(X)

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
class MockDataset(_MolGraphDatasetMixin, MolGraphDataset):
    """A :class:`MockDataset` serves to create a mock empty dataset that passes through all the message passing code.
    This is used when there are no target columns for any of molecule, atom, and/or bond for mixed predictions
    """

    featurizer = SimpleMoleculeMolGraphFeaturizer

    def Y(self) -> np.ndarray:
        """the (scaled) targets of the dataset"""
        return np.array([])

    def __getitem__(self, idx: int) -> Datum:
        return None

    @property
    def cache(self) -> bool:
        return self.__cache

    @cache.setter
    def cache(self, cache: bool = False):
        self.__cache = cache
        self._init_cache()

    def _init_cache(self):
        """initialize the cache"""
        self.mg_cache = (MolGraphCache if self.cache else MolGraphCacheOnTheFly)(
            self.mols, self.V_fs, self.E_fs, self.featurizer
        )

    @cached_property
    def _slices(self) -> list:
        return None

    @property
    def V_fs(self) -> list[np.ndarray]:
        """the raw atom features of the dataset"""
        return np.array([])

    @property
    def E_fs(self) -> list[np.ndarray]:
        """the raw bond features of the dataset"""
        return np.array([])

    @property
    def V_ds(self) -> list[np.ndarray]:
        """the raw atom descriptors of the dataset"""
        return np.array([])

    @property
    def E_ds(self) -> list[np.ndarray]:
        return np.array([])

    @property
    def d_vf(self) -> int:
        """the extra atom feature dimension, if any"""
        return 0

    @property
    def d_ef(self) -> int:
        """the extra bond feature dimension, if any"""
        return 0

    @property
    def d_vd(self) -> int:
        """the extra atom descriptor dimension, if any"""
        return 0

    @property
    def d_ed(self) -> int:
        return 0

    def normalize_inputs(
        self, key: str = "X_d", scaler: StandardScaler | None = None
    ) -> StandardScaler:
        scaler = StandardScaler()
        scaler.mean_ = [0]
        scaler.scale_ = [1]
        return scaler

    def normalize_targets(self, scaler: StandardScaler | None = None) -> StandardScaler:
        scaler = StandardScaler()
        scaler.mean_ = [0]
        scaler.scale_ = [1]
        return scaler


@dataclass
class MoleculeDataset(_MolGraphDatasetMixin, MolGraphDataset):
    """A :class:`MoleculeDataset` composed of :class:`MoleculeDatapoint`\s

    A :class:`MoleculeDataset` produces featurized data for input to a
    :class:`MPNN` model. Typically, data featurization is performed on-the-fly
    and parallelized across multiple workers via the :class:`~torch.utils.data
    DataLoader` class. However, for small datasets, it may be more efficient to
    featurize the data in advance and cache the results. This can be done by
    setting ``MoleculeDataset.cache=True``.

    Parameters
    ----------
    data : Iterable[MoleculeDatapoint]
        the data from which to create a dataset
    featurizer : MoleculeFeaturizer
        the featurizer with which to generate MolGraphs of the molecules
    """

    data: list[MoleculeDatapoint]
    featurizer: Featurizer[Mol, MolGraph] = field(default_factory=SimpleMoleculeMolGraphFeaturizer)

    def __post_init__(self):
        if self.data is None:
            raise ValueError("Data cannot be None!")

        self.reset()
        self.cache = False

    def __getitem__(self, idx: int) -> Datum:
        d = self.data[idx]
        mg = self.mg_cache[idx]

        return Datum(
            mg,
            self.V_ds[idx],
            self.E_ds[idx],
            self.X_d[idx],
            self.Y[idx],
            d.weight,
            d.lt_mask,
            d.gt_mask,
        )

    @property
    def cache(self) -> bool:
        return self.__cache

    @cache.setter
    def cache(self, cache: bool = False):
        self.__cache = cache
        self._init_cache()

    def _init_cache(self):
        """initialize the cache"""
        self.mg_cache = (MolGraphCache if self.cache else MolGraphCacheOnTheFly)(
            self.mols, self.V_fs, self.E_fs, self.featurizer
        )

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
        self._init_cache()

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
        self._init_cache()

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
    def _E_ds(self) -> list[np.ndarray]:
        return [d.E_d for d in self.data]

    @property
    def E_ds(self) -> list[np.ndarray]:
        return self.__E_ds

    @E_ds.setter
    def E_ds(self, E_ds: list[np.ndarray]):
        self._validate_attribute(E_ds, "bond descriptors")

        self.__E_ds = E_ds

    @property
    def d_vf(self) -> int:
        """the extra atom feature dimension, if any"""
        return 0 if self.V_fs[0] is None else self.V_fs[0].shape[1]

    @property
    def d_ef(self) -> int:
        """the extra bond feature dimension, if any"""
        return 0 if self.E_fs[0] is None else self.E_fs[0].shape[1]

    @property
    def d_vd(self) -> int:
        """the extra atom descriptor dimension, if any"""
        return 0 if self.V_ds[0] is None else self.V_ds[0].shape[1]

    @property
    def d_ed(self) -> int:
        return 0 if self.E_ds[0] is None else self.E_ds[0].shape[1]

    def normalize_inputs(
        self, key: str = "X_d", scaler: StandardScaler | None = None
    ) -> StandardScaler:
        VALID_KEYS = {"X_d", "V_f", "E_f", "V_d", "E_d"}

        match key:
            case "X_d":
                X = None if self.d_xd == 0 else self.X_d
            case "V_f":
                X = None if self.d_vf == 0 else np.concatenate(self.V_fs, axis=0)
            case "E_f":
                X = None if self.d_ef == 0 else np.concatenate(self.E_fs, axis=0)
            case "V_d":
                X = None if self.d_vd == 0 else np.concatenate(self.V_ds, axis=0)
            case "E_d":
                X = None if self.d_ed == 0 else np.concatenate(self.E_ds, axis=0)
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
                self.V_fs = [scaler.transform(V_f) if V_f.size > 0 else V_f for V_f in self.V_fs]
            case "E_f":
                self.E_fs = [scaler.transform(E_f) if E_f.size > 0 else E_f for E_f in self.E_fs]
            case "V_d":
                self.V_ds = [scaler.transform(V_d) if V_d.size > 0 else V_d for V_d in self.V_ds]
            case "E_d":
                self.E_ds = [scaler.transform(E_d) if E_d.size > 0 else E_d for E_d in self.E_ds]
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
        self.__E_ds = self._E_ds


@dataclass
class AtomDataset(MoleculeDataset):
    @cached_property
    def _Y(self) -> np.ndarray:
        return np.vstack([d.y for d in self.data])
        # dim = self.data[0].y.shape[1]
        # raw_targets = np.empty((0, dim), float)
        # for d in self.data:
        #     raw_targets = np.vstack([raw_targets, d.y])
        # return raw_targets

    @property
    def Y(self) -> np.ndarray:
        """the (scaled) targets of the dataset"""
        return self.__Y

    @Y.setter
    def Y(self, Y: ArrayLike):
        self.__Y = np.array(Y, float)

    @cached_property
    def _slices(self) -> list:
        slice_indices = []
        index = 0
        for d in self.data:
            slice_indices.extend([index] * d.mol.GetNumAtoms())
            index += 1
        return slice_indices

    @property
    def gt_mask(self) -> np.ndarray:
        return np.vstack([d.gt_mask for d in self.data])
        # dim = self.data[0].gt_mask.shape[1]
        # temp_gt_mask = np.empty((0, dim))
        # for d in self.data:
        #     temp_gt_mask = np.vstack([temp_gt_mask, np.vstack(d.gt_mask)])
        # return temp_gt_mask

    @property
    def lt_mask(self) -> np.ndarray:
        return np.vstack([d.lt_mask for d in self.data])
        # dim = self.data[0].lt_mask.shape[1]
        # temp_lt_mask = np.empty((0, dim))
        # for d in self.data:
        #     temp_lt_mask = np.vstack([temp_lt_mask, np.vstack(d.lt_mask)])
        # return temp_lt_mask

    def __getitem__(self, idx: int) -> Datum:
        d = self.data[idx]
        mg = self.mg_cache[idx]
        slices = self._slices
        ind_first = slices.index(idx)
        ind_last = ind_first + slices.count(idx)

        # TODO: check for E_d?

        return Datum(
            mg,
            self.V_ds[idx],
            self.E_ds[idx],
            self.X_d[idx],
            self.Y[ind_first:ind_last],
            d.weight,
            d.lt_mask,
            d.gt_mask,
        )

    def reset(self):
        """Reset the atom and bond features; atom and extra descriptors; and targets of each
        datapoint to their initial, unnormalized values."""
        super().reset()
        self.__Y = self._Y


@dataclass
class BondDataset(AtomDataset):
    @cached_property
    def _slices(self) -> list:
        slice_indices = []
        index = 0
        for d in self.data:
            slice_indices.extend([index] * d.mol.GetNumBonds())
            index += 1
        return slice_indices


@dataclass(repr=False, eq=False)
class MolAtomBondDataset(_MolGraphDatasetMixin, MolGraphDataset):
    """A :class:`MulticomponentDataset` is a :class:`Dataset` composed of parallel
    :class:`MoleculeDatasets` and :class:`ReactionDataset`\s"""

    mol_dataset: MoleculeDataset | MockDataset
    atom_dataset: AtomDataset | MockDataset
    bond_dataset: BondDataset | MockDataset

    def __post_init__(self):
        self.datasets = [self.mol_dataset, self.atom_dataset, self.bond_dataset]

    def __len__(self) -> int:
        return len(self.datasets[0])

    @property
    def n_components(self) -> int:
        return len(self.datasets)

    def __getitem__(self, idx: int) -> list[Datum]:
        mixed_list = []
        for dset in self.datasets:
            mixed_list.append(
                Datum(
                    dset[idx].mg,
                    dset[idx].V_d,
                    dset[idx].E_d,
                    dset[idx].x_d,
                    dset[idx].y,
                    dset[idx].weight,
                    dset[idx].lt_mask,
                    dset[idx].gt_mask,
                )
                if dset[idx] is not None
                else None
            )

        return mixed_list

    @property
    def featurizer(self):
        return self.datasets[0].featurizer

    @property
    def smiles(self) -> list[list[str]]:
        return list(zip(*[dset.smiles for dset in self.datasets]))

    @property
    def names(self) -> list[str]:
        return self.datasets[0].names

    @property
    def mols(self) -> list[list[Chem.Mol]]:
        return list(zip(*[dset.mols for dset in self.datasets]))

    def normalize_targets(self, scaler: StandardScaler | None = None) -> StandardScaler:
        return self.datasets[0].normalize_targets(scaler)

    def normalize_inputs(
        self, key: str = "X_d", scaler: list[StandardScaler] | None = None
    ) -> list[StandardScaler]:
        match scaler:
            case None:
                return self.datasets[0].normalize_inputs(key)
            case _:
                return self.datasets[0].normalize_inputs(key, scaler)

    def reset(self):
        return [dset.reset() for dset in self.datasets]

    @property
    def d_xd(self) -> int:
        return self.datasets[0].d_xd

    @property
    def d_vf(self) -> int:
        return self.datasets[0].d_vf

    @property
    def d_ef(self) -> int:
        return self.datasets[0].d_ef

    @property
    def d_vd(self) -> int:
        return self.datasets[0].d_vd

    @property
    def d_ed(self) -> int:
        return self.datasets[0].d_ed


@dataclass
class ReactionDataset(_MolGraphDatasetMixin, MolGraphDataset):
    """A :class:`ReactionDataset` composed of :class:`ReactionDatapoint`\s

    .. note::
        The featurized data provided by this class may be cached, simlar to a
        :class:`MoleculeDataset`. To enable the cache, set ``ReactionDataset
        cache=True``.
    """

    data: list[ReactionDatapoint]
    """the dataset from which to load"""
    featurizer: Featurizer[Rxn, MolGraph] = field(default_factory=CGRFeaturizer)
    """the featurizer with which to generate MolGraphs of the input"""

    def __post_init__(self):
        if self.data is None:
            raise ValueError("Data cannot be None!")

        self.reset()
        self.cache = False

    @property
    def cache(self) -> bool:
        return self.__cache

    @cache.setter
    def cache(self, cache: bool = False):
        self.__cache = cache
        self.mg_cache = (MolGraphCache if cache else MolGraphCacheOnTheFly)(
            self.mols, [None] * len(self), [None] * len(self), self.featurizer
        )

    def __getitem__(self, idx: int) -> Datum:
        d = self.data[idx]
        mg = self.mg_cache[idx]

        return Datum(mg, None, None, self.X_d[idx], self.Y[idx], d.weight, d.lt_mask, d.gt_mask)

    @property
    def smiles(self) -> list[tuple]:
        return [(Chem.MolToSmiles(d.rct), Chem.MolToSmiles(d.pdt)) for d in self.data]

    @property
    def mols(self) -> list[Rxn]:
        return [(d.rct, d.pdt) for d in self.data]

    @property
    def d_vf(self) -> int:
        return 0

    @property
    def d_ef(self) -> int:
        return 0

    @property
    def d_vd(self) -> int:
        return 0

    @property
    def d_ed(self) -> int:
        return 0


@dataclass(repr=False, eq=False)
class MulticomponentDataset(_MolGraphDatasetMixin, Dataset):
    """A :class:`MulticomponentDataset` is a :class:`Dataset` composed of parallel
    :class:`MoleculeDatasets` and :class:`ReactionDataset`\s"""

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
    def names(self) -> list[list[str]]:
        return list(zip(*[dset.names for dset in self.datasets]))

    @property
    def mols(self) -> list[list[Chem.Mol]]:
        return list(zip(*[dset.mols for dset in self.datasets]))

    def normalize_targets(self, scaler: StandardScaler | None = None) -> StandardScaler:
        return self.datasets[0].normalize_targets(scaler)

    def normalize_inputs(
        self, key: str = "X_d", scaler: list[StandardScaler] | None = None
    ) -> list[StandardScaler]:
        RXN_VALID_KEYS = {"X_d"}
        match scaler:
            case None:
                return [
                    dset.normalize_inputs(key)
                    if isinstance(dset, MoleculeDataset) or key in RXN_VALID_KEYS
                    else None
                    for dset in self.datasets
                ]
            case _:
                assert len(scaler) == len(
                    self.datasets
                ), "Number of scalers must match number of datasets!"

                return [
                    dset.normalize_inputs(key, s)
                    if isinstance(dset, MoleculeDataset) or key in RXN_VALID_KEYS
                    else None
                    for dset, s in zip(self.datasets, scaler)
                ]

    def reset(self):
        return [dset.reset() for dset in self.datasets]

    @property
    def d_xd(self) -> list[int]:
        return self.datasets[0].d_xd

    @property
    def d_vf(self) -> list[int]:
        return sum(dset.d_vf for dset in self.datasets)

    @property
    def d_ef(self) -> list[int]:
        return sum(dset.d_ef for dset in self.datasets)

    @property
    def d_vd(self) -> list[int]:
        return sum(dset.d_vd for dset in self.datasets)

    @property
    def d_ed(self) -> list[int]:
        return sum(dset.d_ed for dset in self.datasets)
