from argparse import Namespace
from copy import deepcopy
import csv
import math
import os
import pickle
from pprint import pprint

import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH
import hpbandster.core.nameserver as hpns
import hpbandster.core.result as hpres
from hpbandster.core.worker import Worker
from hpbandster.optimizers import BOHB
from utils import StandardScaler
from torch.optim import Adam
from tqdm import trange

from model import build_model
from nn_utils import NoamLR, param_count
from parsing import get_parser, modify_args
from train_utils import train, evaluate
from utils import get_data, get_loss_func, get_metric_func, split_data


class MPNWorker(Worker):
    def __init__(self, args: Namespace, **kwargs):
        super(MPNWorker, self).__init__(**kwargs)

        self.args = args
        self.sign = -1 if args.dataset_type == 'classification' else 1

        print('Loading data')
        data = get_data(self.args.data_path)
        if args.separate_test_set:
            self.train_data, self.val_data, _ = split_data(data, sizes=(0.8, 0.2, 0.0), seed=args.seed)
            self.test_data = get_data(args.separate_test_set, args.dataset_type, num_bins=args.num_bins)
        else:
            self.train_data, self.val_data, self.test_data = split_data(data, seed=args.seed)
        self.num_tasks = len(data[0][1])

        print('Train size = {}, val size = {}, test size = {}'.format(
            len(self.train_data),
            len(self.val_data),
            len(self.test_data))
        )
        print('Number of tasks = {}'.format(self.num_tasks))

        # Initialize scaler which subtracts mean and divides by standard deviation for regression datasets
        if self.args.dataset_type == 'regression':
            print('Fitting scaler')
            train_labels = list(zip(*self.train_data))[1]
            self.scaler = StandardScaler().fit(train_labels)
        else:
            self.scaler = None

        # Get loss and metric functions
        self.loss_func = get_loss_func(self.args.dataset_type)
        self.metric_func = get_metric_func(self.args.metric)

    def compute(self,
                config: dict,
                budget: int,
                *args,
                **kwargs) -> dict:
        """
        Trains a model using the provided configuration and budget.

        :param config: The configuration to use when building/training the model.
        :param budget: The number of epochs to run.
        :param args: Other arguments (not used).
        :param kwargs: Keyword arguments (not used).
        :return: A dictionary containing the best validation loss and other relevant info.
        """
        pprint(config)

        # Update args with config
        args = deepcopy(self.args)
        for key, value in config.items():
            if hasattr(args, key):
                setattr(args, key, value)

        # Build model
        model = build_model(self.num_tasks, args)
        print(model)
        num_params = param_count(model)
        print('Number of parameters = {:,}'.format(num_params))
        if args.cuda:
            print('Moving model to cuda')
            model = model.cuda()

        # Optimizer and learning rate scheduler
        optimizer = Adam(model.parameters(), lr=args.init_lr)
        scheduler = NoamLR(
            optimizer,
            warmup_epochs=args.warmup_epochs,
            total_epochs=args.epochs,
            steps_per_epoch=len(self.train_data) // args.batch_size,
            init_lr=args.init_lr,
            max_lr=args.max_lr,
            final_lr=args.final_lr
        )

        n_iter = 0
        for _ in trange(math.ceil(budget)):
            print("Learning rate = {:.3e}".format(scheduler.get_lr()[0]))

            n_iter = train(
                model=model,
                data=self.train_data,
                loss_func=self.loss_func,
                optimizer=optimizer,
                scheduler=scheduler,
                args=args,
                n_iter=n_iter,
                chunk_names=(args.num_chunks > 1)
            )
            val_score = evaluate(
                model=model,
                data=self.val_data,
                metric_func=self.metric_func,
                args=args,
                scaler=self.scaler
            )

            print('Validation {} = {:.3f}'.format(args.metric, val_score))

        # Compute losses
        scores = {
            split: evaluate(
                model=model,
                data=data,
                metric_func=self.metric_func,
                args=args,
                scaler=self.scaler
            ) for split, data in [('train', self.train_data),
                                  ('val', self.val_data),
                                  ('test', self.test_data)]
        }

        return {
            'loss': self.sign * scores['val'],  # BOHB optimizer tries to minimize loss
            'info': {
                'train_loss': scores['train'],
                'val_loss': scores['val'],
                'test_loss': scores['test'],
                'num_params': num_params
            }
        }

    @staticmethod
    def get_configspace():
        """
        Build configuration space.

        :return: A ConfigSpace object containing the configurations to try.
        """
        cs = CS.ConfigurationSpace()

        cs.add_hyperparameters([
            CSH.UniformIntegerHyperparameter('hidden_size', lower=150, upper=1200),
            CSH.UniformIntegerHyperparameter('depth', lower=2, upper=6),
            CSH.CategoricalHyperparameter('master_node', choices=[True, False]),
            CSH.UniformIntegerHyperparameter('master_dim', lower=100, upper=1200)
        ])

        return cs


def optimize_hyperparameters(args: Namespace):
    # Save intermediate results
    result_logger = hpres.json_result_logger(directory=args.results_dir, overwrite=True)

    # Start HpBandSter server
    NS = hpns.NameServer(run_id='example1', host='127.0.0.1', port=args.port)
    NS.start()

    # Create HpBandSter worker which runs model
    worker = MPNWorker(args, run_id='example1', nameserver='127.0.0.1')
    worker.run(background=True)

    # Create HpBandSter BOHB optimizer
    bohb = BOHB(
        configspace=worker.get_configspace(),
        run_id='example1',
        nameserver='127.0.0.1',
        result_logger=result_logger,
        eta=args.eta,
        min_budget=args.min_budget,
        max_budget=args.max_budget
    )
    res = bohb.run(n_iterations=args.n_iterations)

    # Print results
    id2config = res.get_id2config_mapping()
    incumbent = res.get_incumbent_id()

    print('Best found configuration:', id2config[incumbent]['config'])
    print('Performance of best configuration:', res.get_runs_by_id(incumbent)[-1]['info'])
    print('A total of %i unique configurations where sampled.' % len(id2config.keys()))
    print('A total of %i runs were executed.' % len(res.get_all_runs()))
    print('Total budget corresponds to %.1f full function evaluations.' % (
                sum([r.budget for r in res.get_all_runs()]) / args.max_budget))

    # Save all results with pickle
    with open(os.path.join(args.results_dir, 'results.pkl'), 'wb') as f:
        pickle.dump(res, f)

    # Save best result for each config in CSV format
    results = []
    for id, config in id2config.items():
        for info in res.get_runs_by_id(id):
            results.append(
                dict(
                    **config['config'],
                    **info['info'],
                    budget=info['budget']
                )
            )
    results.sort(key=lambda r: r['val_loss'])

    with open(os.path.join(args.results_dir, 'results.csv'), 'w') as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    bohb.shutdown(shutdown_workers=True)
    NS.shutdown()


if __name__ == '__main__':
    parser = get_parser()
    parser.add_argument('--results_dir', type=str,
                        help='Path to directory where results will be saved')
    parser.add_argument('--port', type=int, default=9090,
                        help='Port for HpBandSter to use')
    parser.add_argument('--min_budget', type=int, default=3,
                        help='Minimum budget (number of iterations during training) to use')
    parser.add_argument('--max_budget', type=int, default=30,
                        help='Maximum budget (number of iterations during training) to use')
    parser.add_argument('--eta', type=int, default=2,
                        help='Factor by which to cut number of trials (1/eta trials remain)')
    parser.add_argument('--n_iterations', type=int, default=4,
                        help='Number of iterations of BOHB algorithm')
    args = parser.parse_args()

    modify_args(args)

    os.makedirs(args.results_dir, exist_ok=True)

    optimize_hyperparameters(args)
