from typing import List

import numpy as np


class StandardScaler:
    def __init__(self, means: np.ndarray = None, stds: np.ndarray = None):
        """Initialize StandardScaler, optionally with means and standard deviations precomputed."""
        self.means = means
        self.stds = stds

    def fit(self, X: List[List[float]]) -> 'StandardScaler':
        """
        Learns means and standard deviations across the 0-th axis.

        :param X: A list of lists of floats.
        :return: The fitted StandardScaler.
        """
        X = np.array(X).astype(float)
        self.means = np.nanmean(X, axis=0)
        self.stds = np.nanstd(X, axis=0)
        self.stds = np.where(self.stds == 0, self.stds, np.ones(self.stds.shape))

        return self

    def transform(self, X: List[List[float]]):
        """
        Transforms the data by subtracting the means and dividing by the standard deviations.

        :param X: A list of lists of floats.
        :return: The transformed data.
        """
        X = np.array(X).astype(float)
        transformed_with_nan = (X - self.means) / self.stds
        transformed_with_none = np.where(np.isnan(transformed_with_nan), None, transformed_with_nan)

        return transformed_with_none.astype(float)

    def inverse_transform(self, X: List[List[float]]):
        """
        Performs the inverse transformation by multiplying by the standard deviations and adding the means.

        :param X: A list of lists of floats.
        :return: The inverse transformed data.
        """
        X = np.array(X).astype(float)
        transformed_with_nan = X * self.stds + self.means
        transformed_with_none = np.where(np.isnan(transformed_with_nan), None, transformed_with_nan)

        return transformed_with_none.astype(float)
