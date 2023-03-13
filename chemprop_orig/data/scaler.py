from typing import Any, List, Optional

import numpy as np


class StandardScaler:
    """A :class:`StandardScaler` normalizes the features of a dataset.

    When it is fit on a dataset, the :class:`StandardScaler` learns the mean and standard deviation across the 0th axis.
    When transforming a dataset, the :class:`StandardScaler` subtracts the means and divides by the standard deviations.
    """

    def __init__(self, means: np.ndarray = None, stds: np.ndarray = None, replace_nan_token: Any = None):
        """
        :param means: An optional 1D numpy array of precomputed means.
        :param stds: An optional 1D numpy array of precomputed standard deviations.
        :param replace_nan_token: A token to use to replace NaN entries in the features.
        """
        self.means = means
        self.stds = stds
        self.replace_nan_token = replace_nan_token

    def fit(self, X: List[List[Optional[float]]]) -> 'StandardScaler':
        """
        Learns means and standard deviations across the 0th axis of the data :code:`X`.

        :param X: A list of lists of floats (or None).
        :return: The fitted :class:`StandardScaler` (self).
        """
        X = np.array(X).astype(float)
        self.means = np.nanmean(X, axis=0)
        self.stds = np.nanstd(X, axis=0)
        self.means = np.where(np.isnan(self.means), np.zeros(self.means.shape), self.means)
        self.stds = np.where(np.isnan(self.stds), np.ones(self.stds.shape), self.stds)
        self.stds = np.where(self.stds == 0, np.ones(self.stds.shape), self.stds)

        return self

    def transform(self, X: List[List[Optional[float]]]) -> np.ndarray:
        """
        Transforms the data by subtracting the means and dividing by the standard deviations.

        :param X: A list of lists of floats (or None).
        :return: The transformed data with NaNs replaced by :code:`self.replace_nan_token`.
        """
        X = np.array(X).astype(float)
        transformed_with_nan = (X - self.means) / self.stds
        transformed_with_none = np.where(np.isnan(transformed_with_nan), self.replace_nan_token, transformed_with_nan)

        return transformed_with_none

    def inverse_transform(self, X: List[List[Optional[float]]]) -> np.ndarray:
        """
        Performs the inverse transformation by multiplying by the standard deviations and adding the means.

        :param X: A list of lists of floats.
        :return: The inverse transformed data with NaNs replaced by :code:`self.replace_nan_token`.
        """
        X = np.array(X).astype(float)
        transformed_with_nan = X * self.stds + self.means
        transformed_with_none = np.where(np.isnan(transformed_with_nan), self.replace_nan_token, transformed_with_nan)

        return transformed_with_none

class AtomBondScaler(StandardScaler):
    """A :class:`AtomBondScaler` normalizes the features of a dataset.

    When it is fit on a dataset, the :class:`AtomBondScaler` learns the mean and standard deviation across the 0th axis.
    When transforming a dataset, the :class:`AtomBondScaler` subtracts the means and divides by the standard deviations.
    """

    def __init__(self, means: np.ndarray = None, stds: np.ndarray = None, replace_nan_token: Any = None, n_atom_targets = None, n_bond_targets = None):
        super().__init__(means, stds, replace_nan_token)
        self.n_atom_targets = n_atom_targets
        self.n_bond_targets = n_bond_targets

    def fit(self, X: List[List[Optional[float]]]) -> 'AtomBondScaler':
        scalers = []
        for i in range(self.n_atom_targets):
            scaler = StandardScaler().fit(X[i])
            scalers.append(scaler)
        for i in range(self.n_bond_targets):
            scaler = StandardScaler().fit(X[i+self.n_atom_targets])
            scalers.append(scaler)

        self.means = np.array([s.means for s in scalers])
        self.stds = np.array([s.stds for s in scalers])

        return self

    def transform(self, X: List[List[Optional[float]]]) -> List[np.ndarray]:
        """
        Transforms the data by subtracting the means and dividing by the standard deviations.

        :param X: A list of lists of floats (or None).
        :return: The transformed data with NaNs replaced by :code:`self.replace_nan_token`.
        """
        transformed_results = []
        for i in range(self.n_atom_targets):
            Xi = np.array(X[i]).astype(float)
            transformed_with_nan = (Xi - self.means[i]) / self.stds[i]
            transformed_with_none = np.where(np.isnan(transformed_with_nan), self.replace_nan_token, transformed_with_nan)
            transformed_results.append(transformed_with_none.tolist())
        for i in range(self.n_bond_targets):
            Xi = np.array(X[i+self.n_atom_targets]).astype(float)
            transformed_with_nan = (Xi - self.means[i+self.n_atom_targets]) / self.stds[i+self.n_atom_targets]
            transformed_with_none = np.where(np.isnan(transformed_with_nan), self.replace_nan_token, transformed_with_nan)
            transformed_results.append(transformed_with_none.tolist())

        return transformed_results

    def inverse_transform(self, X: List[List[Optional[float]]]) -> List[np.ndarray]:
        """
        Performs the inverse transformation by multiplying by the standard deviations and adding the means.

        :param X: A list of lists of floats.
        :return: The inverse transformed data with NaNs replaced by :code:`self.replace_nan_token`.
        """
        transformed_results = []
        for i in range(self.n_atom_targets):
            Xi = np.array(X[i]).astype(float)
            transformed_with_nan = Xi * self.stds[i] + self.means[i]
            transformed_with_none = np.where(np.isnan(transformed_with_nan), self.replace_nan_token, transformed_with_nan)
            transformed_results.append(transformed_with_none.tolist())
        for i in range(self.n_bond_targets):
            Xi = np.array(X[i+self.n_atom_targets]).astype(float)
            transformed_with_nan = Xi * self.stds[i+self.n_atom_targets] + self.means[i+self.n_atom_targets]
            transformed_with_none = np.where(np.isnan(transformed_with_nan), self.replace_nan_token, transformed_with_nan)
            transformed_results.append(transformed_with_none.tolist())

        return transformed_results
