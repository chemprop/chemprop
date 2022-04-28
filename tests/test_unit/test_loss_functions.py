"""Chemprop unit tests for chemprop/train/loss_functions.py"""
from unittest import TestCase
from types import SimpleNamespace

import numpy as np
import torch

from chemprop.train.loss_functions import (
    bounded_mse_loss,
    dirichlet_class_loss,
    evidential_loss,
    get_loss_func,
    normal_mve,
)


class TestGetLossFunction(TestCase):
    """
    Tests get_loss_func function.
    """

    def test_supported(self):
        supported_loss_functions = {
            "regression": ["mse", "bounded_mse", "mve", "evidential"],
            "classification": ["binary_cross_entropy", "mcc", "dirichlet"],
            "multiclass": ["cross_entropy", "mcc", "dirichlet"],
            "spectra": ["sid", "wasserstein"],
        }
        for dtype in supported_loss_functions:
            for fx in supported_loss_functions[dtype]:
                args = SimpleNamespace(dataset_type=dtype, loss_function=fx)
                self.assertIsNotNone(get_loss_func(args=args))

    def test_unsupported(self):
        with self.assertRaises(ValueError):
            args = SimpleNamespace(
                dataset_type="regression", loss_function="dummy_loss"
            )
            self.assertIsNotNone(get_loss_func(args=args))


class TestBoundedMSE(TestCase):
    """
    Tests the bounded_mse loss function
    """

    def setUp(self):
        self.preds = torch.tensor([[-3, 2], [1, -1]], dtype=float)
        self.targets = torch.zeros([2, 2], dtype=float)

    def test_no_inequality(self):
        lt_targets = torch.zeros([2, 2], dtype=bool)
        gt_targets = torch.zeros([2, 2], dtype=bool)
        loss = bounded_mse_loss(self.preds, self.targets, lt_targets, gt_targets)
        self.assertEqual(loss.sum(), 15)

    def test_greater_thans(self):
        lt_targets = torch.zeros([2, 2], dtype=bool)
        gt_targets = torch.ones([2, 2], dtype=bool)
        loss = bounded_mse_loss(self.preds, self.targets, lt_targets, gt_targets)
        self.assertEqual(loss.sum(), 10)

    def test_less_thans(self):
        lt_targets = torch.ones([2, 2], dtype=bool)
        gt_targets = torch.zeros([2, 2], dtype=bool)
        loss = bounded_mse_loss(self.preds, self.targets, lt_targets, gt_targets)
        self.assertEqual(loss.sum(), 5)


class TestMVE(TestCase):
    """
    Tests the normal_mve loss function.
    """

    def test_simple(self):
        preds = torch.tensor([[0, 1]], dtype=float)
        targets = torch.zeros([1, 1])
        loss = normal_mve(preds, targets)
        self.assertAlmostEqual(
            0.3989, np.exp(-1 * loss).item(), places=4
        )  # pdf at zero stdev 1


class TestDirichlet(TestCase):
    """
    Tests the dirichlet_class_loss function.
    """

    def test_simple(self):
        alphas = torch.tensor([[2, 2]])
        target_labels = torch.ones([1, 1])
        loss = dirichlet_class_loss(alphas, target_labels)
        self.assertAlmostEqual(0.6, loss.item())

    def test_lambda(self):
        alphas = torch.tensor([[2, 2]])
        target_labels = torch.ones([1, 1])
        loss = dirichlet_class_loss(alphas, target_labels, lam=0.2)
        self.assertAlmostEqual(0.63862943, loss.item())

    def test_wrong_input_size(self):
        with self.assertRaises(RuntimeError):
            alphas = torch.ones([2, 2])
            target_labels = torch.ones([2, 2])
            loss = dirichlet_class_loss(alphas, target_labels)


class TestEvidential(TestCase):
    """
    Tests the evidential_loss function
    """

    def test_simple(self):
        preds = torch.tensor([[2, 2, 2, 2]])
        targets = torch.ones([1, 1])
        loss = evidential_loss(preds, targets)
        self.assertAlmostEqual(1.56893861, loss.item())

    def test_lambda(self):
        preds = torch.tensor([[2, 2, 2, 2]])
        targets = torch.ones([1, 1])
        loss = evidential_loss(preds, targets, lam=0.2)
        self.assertAlmostEqual(2.768938541, loss.item())

    def test_wrong_input_size(self):
        with self.assertRaises(RuntimeError):
            preds = torch.ones([2, 2])
            targets = torch.ones([2, 2])
            loss = evidential_loss(preds, targets)
