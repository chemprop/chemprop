import unittest
import numpy as np
from chemprop.utils import multitask_mean


class TestMean(unittest.TestCase):
    def _test_multitask_mean(self, metric_name: str):
        scores = np.random.random(5)
        scores[4] = np.nan
        metric = multitask_mean(scores, metric_name)
        self.assertTrue(np.isfinite(metric).all())

    def test_classification_multitask_mean(self):
        self._test_multitask_mean('auc')

    def test_regression_multitask_mean(self):
        self._test_multitask_mean('mae')
