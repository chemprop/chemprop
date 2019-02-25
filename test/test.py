"""Very basic tests to check that nothing crashes. Not 100% coverage but covers what we care about."""

from argparse import ArgumentParser
from copy import deepcopy
import os
import sys
from tempfile import TemporaryDirectory, NamedTemporaryFile
import unittest

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from chemprop.data.utils import get_data
from chemprop.features import clear_cache, get_available_features_generators
from chemprop.parsing import add_train_args, modify_train_args, add_predict_args, modify_predict_args
from chemprop.train import cross_validate, make_predictions
from chemprop.utils import create_logger
from hyperparameter_optimization import grid_search
from scripts.avg_dups import average_duplicates
from scripts.overlap import overlap
from scripts.save_features import generate_and_save_features
from scripts.similarity import scaffold_similarity, morgan_similarity


# TODO hyperopt


class TestScripts(unittest.TestCase):
    def tearDown(self):
        clear_cache()

    def test_avg_dups(self):
        try:
            parser = ArgumentParser()
            parser.add_argument('--data_path', type=str,
                                help='Path to data CSV file')
            parser.add_argument('--save_path', type=str,
                                help='Path where average data CSV file will be saved')
            args = parser.parse_args([])
            args.data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'delaney_toy.csv')
            args.save_path = NamedTemporaryFile().name
            average_duplicates(args)
            os.remove(args.save_path)
        except:
            self.fail('avg_dups')
    
    def test_overlap(self):
        try:
            parser = ArgumentParser()
            parser.add_argument('--data_path_1', type=str,
                                help='Path to first data CSV file')
            parser.add_argument('--data_path_2', type=str,
                                help='Path to second data CSV file')
            parser.add_argument('--compound_names_1', action='store_true', default=False,
                                help='Whether data_path_1 has compound names in addition to smiles')
            parser.add_argument('--compound_names_2', action='store_true', default=False,
                                help='Whether data_path_2 has compound names in addition to smiles')
            parser.add_argument('--save_intersection_path', type=str, default=None,
                                help='Path to save intersection at; labeled with data_path 1 header')
            parser.add_argument('--save_difference_path', type=str, default=None,
                                help='Path to save molecules in dataset 1 that are not in dataset 2; labeled with data_path 1 header')
            args = parser.parse_args([])
            args.data_path_1 = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'delaney_toy.csv')
            args.data_path_2 = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'delaney_toy.csv')

            overlap(args)
        except:
            self.fail('overlap')
    
    def test_save_features(self):
        try:
            parser = ArgumentParser()
            parser.add_argument('--data_path', type=str,
                                help='Path to data CSV')
            parser.add_argument('--features_generator', type=str,
                                choices=get_available_features_generators(),
                                help='Type of features to generate')
            parser.add_argument('--save_path', type=str,
                                help='Path to .npz file where features will be saved as a numpy compressed archive')
            parser.add_argument('--save_frequency', type=int, default=10000,
                                help='Frequency with which to save the features')
            parser.add_argument('--restart', action='store_true', default=False,
                                help='Whether to not load partially complete featurization and instead start from scratch')
            parser.add_argument('--max_data_size', type=int,
                                help='Maximum number of data points to load')
            parser.add_argument('--sequential', action='store_true', default=False,
                                help='Whether to run sequentially rather than in parallel')
            args = parser.parse_args([])
            args.data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'delaney_toy.csv')
            args.save_path = NamedTemporaryFile(suffix='.npz').name
            args.restart = True
            args.features_generator = 'morgan_count'

            generate_and_save_features(args)

            os.remove(args.save_path)
        except:
            self.fail('save_features')
    
    def test_similarity(self):
        try:
            parser = ArgumentParser()
            parser.add_argument('--data_path_1', type=str,
                                help='Path to first data CSV file')
            parser.add_argument('--data_path_2', type=str,
                                help='Path to second data CSV file')
            parser.add_argument('--compound_names_1', action='store_true', default=False,
                                help='Whether data_path_1 has compound names in addition to smiles')
            parser.add_argument('--compound_names_2', action='store_true', default=False,
                                help='Whether data_path_2 has compound names in addition to smiles')
            parser.add_argument('--radius', type=int, default=3,
                                help='Radius of Morgan fingerprint')
            parser.add_argument('--sample_rate', type=float, default=1.0,
                                help='Rate at which to sample pairs of molecules for Morgan similarity (to reduce time)')
            args = parser.parse_args([])

            args.data_path_1 = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'delaney_toy.csv')
            args.data_path_2 = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'delaney_toy.csv')

            data_1 = get_data(path=args.data_path_1, use_compound_names=args.compound_names_1)
            data_2 = get_data(path=args.data_path_2, use_compound_names=args.compound_names_2)

            scaffold_similarity(data_1.smiles(), data_2.smiles())
            morgan_similarity(data_1.smiles(), data_2.smiles(), args.radius, args.sample_rate)
        except:
            self.fail('similarity')


class TestTrain(unittest.TestCase):
    def setUp(self):
        parser = ArgumentParser()
        add_train_args(parser)
        args = parser.parse_args([])
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
        clear_cache()

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
    
    def test_features_path(self):
        try:
            self.args.features_path = [os.path.join(os.path.dirname(os.path.abspath(__file__)), 'delaney_toy_features.npz')]
            self.args.no_features_scaling = True
            modify_train_args(self.args)
            cross_validate(self.args, self.logger)
        except:
            self.fail('features_path')
    
    def test_separate_val_test(self):
        try:
            self.args.separate_val_set = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'delaney_toy.csv')
            self.args.separate_test_set = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'delaney_toy.csv')
            self.args.features_path = [os.path.join(os.path.dirname(os.path.abspath(__file__)), 'delaney_toy_features.npz')]
            self.args.separate_val_set_features = [os.path.join(os.path.dirname(os.path.abspath(__file__)), 'delaney_toy_features.npz')]
            self.args.separate_test_set_features = [os.path.join(os.path.dirname(os.path.abspath(__file__)), 'delaney_toy_features.npz')]
            self.args.no_features_scaling = True
            modify_train_args(self.args)
            cross_validate(self.args, self.logger)
        except:
            self.fail('separate_val_test')
    
    def test_predetermined_split(self):
        try:
            self.args.split_type = 'predetermined'
            self.args.folds_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'delaney_toy_folds.pkl')
            self.args.val_fold_index = 1
            self.args.test_fold_index = 2
            modify_train_args(self.args)
            cross_validate(self.args, self.logger)
        except:
            self.fail('predetermined_split')
    
    def test_config(self):
        try:
            self.args.config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.json')
            modify_train_args(self.args)
            cross_validate(self.args, self.logger)
        except:
            self.fail('config')


class TestPredict(unittest.TestCase):
    def setUp(self):
        parser = ArgumentParser()
        add_train_args(parser)
        args = parser.parse_args([])
        args.data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'delaney_toy.csv')
        args.dataset_type = 'regression'
        args.batch_size = 2
        args.hidden_size = 5
        args.epochs = 1
        args.quiet = True
        self.temp_dir = TemporaryDirectory()
        args.save_dir = self.temp_dir.name
        logger = create_logger(name='train', save_dir=args.save_dir, quiet=args.quiet)
        modify_train_args(args)
        cross_validate(args, logger)
        clear_cache()

        parser = ArgumentParser()
        add_predict_args(parser)
        args = parser.parse_args([])
        args.batch_size = 2
        args.checkpoint_dir = self.temp_dir.name
        args.preds_path = NamedTemporaryFile().name
        args.test_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'delaney_toy_smiles.csv')
        self.args = args
    
    def tearDown(self):
        self.temp_dir.cleanup()
        os.remove(self.args.preds_path)
        self.args = None
        clear_cache()
    
    def test_predict(self):
        try:
            modify_predict_args(self.args)
            make_predictions(self.args)
        except:
            self.fail('predict')
    
    def test_predict_compound_names(self):
        try:
            self.args.test_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'delaney_toy_smiles_names.csv')
            self.args.compound_names = True
            modify_predict_args(self.args)
            make_predictions(self.args)
        except:
            self.fail('predict_compound_names')


class TestHyperopt(unittest.TestCase):
    def test_hyperopt(self):
        try:
            parser = ArgumentParser()
            add_train_args(parser)
            parser.add_argument('--num_iters', type=int, default=20,
                                help='Number of hyperparameter choices to try')
            parser.add_argument('--config_save_path', type=str,
                                help='Path to .json file where best hyperparameter settings will be written')
            parser.add_argument('--log_dir', type=str,
                                help='(Optional) Path to a directory where all results of the hyperparameter optimization will be written')
            args = parser.parse_args([])
            args.data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'delaney_toy.csv')
            args.dataset_type = 'regression'
            args.batch_size = 2
            args.hidden_size = 5
            args.epochs = 1
            args.quiet = True
            temp_file = NamedTemporaryFile()
            args.config_save_path = temp_file.name
            args.num_iters = 3
            modify_train_args(args)

            grid_search(args)
            clear_cache()
        except:
            self.fail('hyperopt')


if __name__ == '__main__':
    unittest.main()
