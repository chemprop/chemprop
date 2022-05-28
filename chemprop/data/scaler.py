from __future__ import annotations

from typing import Any, List, Optional

import numpy as np


class StandardScaler:
    """A :class:`StandardScaler` normalizes the features of a dataset.

    When it is fit on a dataset, the :class:`StandardScaler` learns the mean and standard deviation across the 0th axis.
    When transforming a dataset, the :class:`StandardScaler` subtracts the means and divides by the standard deviations.
    """

    def __init__(
        self, means: np.ndarray = None, stds: np.ndarray = None, replace_nan_token: Any = None
    ):
        """
        :param means: An optional 1D numpy array of precomputed means.
        :param stds: An optional 1D numpy array of precomputed standard deviations.
        :param replace_nan_token: A token to use to replace NaN entries in the features.
        """
        self.means = means
        self.stds = stds
        self.replace_nan_token = replace_nan_token

    def fit(self, X: List[List[Optional[float]]]) -> StandardScaler:
        """
        Learns means and standard deviations across the 0th axis of the data :code:`X`.

        :param X: A list of lists of floats (or None).
        :return: The fitted :class:`StandardScaler` (self).
        """
        X = np.array(X).astype(float)

        self.means = np.nanmean(X, axis=0)
        self.means[np.isnan(self.means)] = 0
        
        self.stds = np.nanstd(X, axis=0)
        self.stds[np.isnan(self.stds)] = 1
        self.stds[self.stds == 0] = 1

        return self

    def transform(self, X: List[List[Optional[float]]]) -> np.ndarray:
        """
        Transforms the data by subtracting the means and dividing by the standard deviations.

        :param X: A list of lists of floats (or None).
        :return: The transformed data with NaNs replaced by :code:`self.replace_nan_token`.
        """
        X = np.array(X).astype(float)
        X_t = (X - self.means) / self.stds
        X_t[np.isnan(X_t)] = self.replace_nan_token

        return X_t

    def inverse_transform(self, X_t: List[List[Optional[float]]]) -> np.ndarray:
        """
        Performs the inverse transformation by multiplying by the standard deviations and adding the means.

        :param X: A list of lists of floats.
        :return: The inverse transformed data with NaNs replaced by :code:`self.replace_nan_token`.
        """
        X_t = np.array(X_t).astype(float)
        X = X_t * self.stds + self.means
        X[np.isnan(X)] = self.replace_nan_token

        return X
