from argparse import ArgumentParser
import os
import logging
from copy import deepcopy
from tempfile import TemporaryDirectory
import unittest

import sys
sys.path.append('../')

from chemprop.parsing import add_train_args, modify_train_args
from chemprop.utils import create_logger
from chemprop.train import cross_validate

# very basic tests to check that nothing crashes. not 100% coverage but covers basically everything we care about

# TODO features_path, separate val/test with features, folds with predetermined splits, config path

class TestTrain(unittest.TestCase):

    def setUp(self):
        parser = ArgumentParser()
        add_train_args(parser)
        args = parser.parse_args()
        args.data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'delaney_toy.csv')
        args.dataset_type = 'regression'
        args.batch_size = 2
        args.hidden_size = 5
        args.epochs = 1
        args.quiet = True
        self.args = args
        logger = create_logger(name='train', save_dir=args.save_dir, quiet=args.quiet)
        self.logger = logger
    
    def tearDown(self):
        self.args = None
        self.logger = None

    def test_regression_default(self):
        try:
            modify_train_args(self.args)
            cross_validate(self.args, self.logger)
        except:
            self.fail('regression_default')
    
    def test_classification_multiclass_default(self):
        try:
            self.args.data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tox21_toy.csv')
            self.args.dataset_type = 'classification'
            modify_train_args(self.args)
            cross_validate(self.args, self.logger)
        except:
            self.fail('classification_default')
    
    def test_rdkit_2d_features(self):
        try:
            self.args.features_generator = ['rdkit_2d_normalized']
            self.args.no_features_scaling = True
            modify_train_args(self.args)
            cross_validate(self.args, self.logger)
        except:
            self.fail('rdkit_2d_features')
    
    def test_rdkit_2d_features_unnormalized(self):
        try:
            self.args.features_generator = ['rdkit_2d']
            modify_train_args(self.args)
            cross_validate(self.args, self.logger)
        except:
            self.fail('rdkit_2d_features_unnormalized')

    def test_features_only(self):
        try:
            self.args.features_generator = ['morgan']
            self.features_only = True
            modify_train_args(self.args)
            cross_validate(self.args, self.logger)
        except:
            self.fail('features_only')
    
    def test_checkpoint(self):
        try:
            args_copy = deepcopy(self.args)
            temp_dir = TemporaryDirectory()
            self.args.save_dir = temp_dir.name
            modify_train_args(self.args)
            cross_validate(self.args, self.logger)
            args_copy.checkpoint_dir = temp_dir.name
            args_copy.test = True
            modify_train_args(args_copy)
            cross_validate(args_copy, self.logger)
        except:
            self.fail('checkpoint')
    
    def test_save_smiles_splits(self):
        try:
            self.args.save_smiles_splits = True
            modify_train_args(self.args)
            cross_validate(self.args, self.logger)
        except:
            self.fail('save smiles splits')
    
    def test_scaffold(self):
        try:
            self.args.split_type = 'scaffold_balanced'
            modify_train_args(self.args)
            cross_validate(self.args, self.logger)
        except:
            self.fail('scaffold')
    
    def test_no_cuda(self):
        try:
            self.args.no_cuda = True
            modify_train_args(self.args)
            cross_validate(self.args, self.logger)
        except:
            self.fail('no_cuda')
    
    def test_show_individual_scores(self):
        try:
            self.args.show_individual_scores = True
            modify_train_args(self.args)
            cross_validate(self.args, self.logger)
        except:
            self.fail('show_individual_scores')
    
    def test_no_cache(self):
        try:
            self.args.no_cache = True
            modify_train_args(self.args)
            cross_validate(self.args, self.logger)
        except:
            self.fail('no_cache')

    def test_num_folds_ensemble(self):
        try:
            self.args.num_folds = 2
            self.args.ensemble_size = 2
            modify_train_args(self.args)
            cross_validate(self.args, self.logger)
        except:
            self.fail('num_folds_ensemble')

    def test_bias(self):
        try:
            self.args.bias = True
            modify_train_args(self.args)
            cross_validate(self.args, self.logger)
        except:
            self.fail('bias')

    def test_activation_prelu(self):
        try:
            self.args.activation = 'PReLU'
            modify_train_args(self.args)
            cross_validate(self.args, self.logger)
        except:
            self.fail('activation_prelu')

    def test_undirected_messages(self):
        try:
            self.args.undirected = True
            modify_train_args(self.args)
            cross_validate(self.args, self.logger)
        except:
            self.fail('undirected_messages')

    def test_atom_messages(self):
        try:
            self.args.atom_messages = True
            modify_train_args(self.args)
            cross_validate(self.args, self.logger)
        except:
            self.fail('atom_messages')

if __name__ == '__main__':
    unittest.main()