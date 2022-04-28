"""Chemprop unit tests for chemprop/uncertainty/uncertainty_evaluator.py"""
from unittest import TestCase
import numpy as np

from chemprop.uncertainty.uncertainty_evaluator import UncertaintyEvaluator, build_uncertainty_evaluator


class TestBuildEvaluator(TestCase):
    """
    Tests build_uncertainty_evaluator function.
    """

    def test_supported(self):
        metrics = {
            "regression": ["nll", "miscalibration_area", "ence", "spearman"],
            "classification": ["auc", "prc-auc", "accuracy", "f1", "mcc"],
            "multiclass": ["cross_entropy", "accuracy", "f1", "mcc"],
        }
        for dtype in metrics:
            for met in metrics[dtype]:
                evaluator = build_uncertainty_evaluator(
                    met, None, None, dtype, None, None
                )
                self.assertIsInstance(evaluator, UncertaintyEvaluator)

    def test_unsupported(self):
        with self.assertRaises(NotImplementedError):
            evaluator = build_uncertainty_evaluator(
                "nll", None, None, "spectra", "sid", None
            )


class TestNLLRegression(TestCase):
    """
    Tests NLLRegressionEvaluator class
    """

    def test(self):
        evaluator = build_uncertainty_evaluator(
            evaluation_method="nll",
            calibration_method=None,
            uncertainty_method="ensemble",
            dataset_type="regression",
            loss_function="mse",
            calibrator=None,
        )
        preds = [[0]]
        targets = [[0]]
        unc = [[1]]
        nll = evaluator.evaluate(targets, preds, unc)
        self.assertAlmostEqual(0.3989, np.exp(-1 * nll[0]), places=4)


class TestNLLClass(TestCase):
    """
    Tests NLLRegressionEvaluator class
    """

    def test_positive(self):
        evaluator = build_uncertainty_evaluator(
            evaluation_method="nll",
            calibration_method=None,
            uncertainty_method="classification",
            dataset_type="classification",
            loss_function="binary_cross_entropy",
            calibrator=None,
        )
        preds = [[0.8]]
        targets = [[1]]
        unc = [[0.8]]
        nll = evaluator.evaluate(targets, preds, unc)
        self.assertAlmostEqual(0.8, np.exp(-1 * nll[0]), places=4)

    def test_negative(self):
        evaluator = build_uncertainty_evaluator(
            evaluation_method="nll",
            calibration_method=None,
            uncertainty_method="classification",
            dataset_type="classification",
            loss_function="binary_cross_entropy",
            calibrator=None,
        )
        preds = [[0.8]]
        targets = [[0]]
        unc = [[0.8]]
        nll = evaluator.evaluate(targets, preds, unc)
        self.assertAlmostEqual(0.2, np.exp(-1 * nll[0]), places=4)


class TestMiscalibration(TestCase):
    """
    Tests CalibrationAreaEvaluator class
    """

    def test_high(self):
        evaluator = build_uncertainty_evaluator(
            evaluation_method="miscalibration_area",
            calibration_method=None,
            uncertainty_method="ensemble",
            dataset_type="regression",
            loss_function="mse",
            calibrator=None,
        )
        preds = np.zeros([100, 1])
        targets = np.zeros([100, 1])
        unc = np.ones([100, 1])
        area = evaluator.evaluate(targets, preds, unc)
        self.assertAlmostEqual(0.495, area[0])

    def test_low(self):
        evaluator = build_uncertainty_evaluator(
            evaluation_method="miscalibration_area",
            calibration_method=None,
            uncertainty_method="ensemble",
            dataset_type="regression",
            loss_function="mse",
            calibrator=None,
        )
        preds = np.zeros([100, 1])
        targets = np.full([100, 1], 100)
        unc = np.ones([100, 1])
        area = evaluator.evaluate(targets, preds, unc)
        self.assertAlmostEqual(0.495, area[0])


class TestSpearman(TestCase):
    """
    Tests SpearmanEvaluator class
    """

    def test_high(self):
        evaluator = build_uncertainty_evaluator(
            evaluation_method="spearman",
            calibration_method=None,
            uncertainty_method="ensemble",
            dataset_type="regression",
            loss_function="mse",
            calibrator=None,
        )
        preds = np.zeros([100, 1])
        targets = np.arange(1, 101).reshape([100, 1])
        unc = np.arange(1, 101).reshape([100, 1])
        spmn = evaluator.evaluate(targets, preds, unc)
        self.assertAlmostEqual(1, spmn[0])

    def test_low(self):
        evaluator = build_uncertainty_evaluator(
            evaluation_method="spearman",
            calibration_method=None,
            uncertainty_method="ensemble",
            dataset_type="regression",
            loss_function="mse",
            calibrator=None,
        )
        preds = np.zeros([100, 1])
        targets = np.arange(1, 101).reshape([100, 1])
        unc = -1 * np.arange(1, 101).reshape([100, 1])
        spmn = evaluator.evaluate(targets, preds, unc)
        self.assertAlmostEqual(-1, spmn[0])
