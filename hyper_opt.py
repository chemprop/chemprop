from argparse import Namespace
from collections import defaultdict
from copy import deepcopy
import json
import logging
import math
import os
from pprint import pprint
from typing import List

import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH
import hpbandster.core.nameserver as hpns
import hpbandster.core.result as hpres
from hpbandster.core.worker import Worker
from hpbandster.optimizers import BOHB
import numpy as np

from chemprop.parsing import parse_hyper_opt_args
from chemprop.train import run_training


def load_sorted_results(results_dir: str) -> List[dict]:
    """
    Loads results and corresponding configs sorted from smallest to greatest loss.

    :param results_dir: Directory containing configs.json and results.json.
    :return: A list of configs/results in sorted order.
    """
    # Get results
    id_to_results = defaultdict(list)
    with open(os.path.join(results_dir, 'results.json'), 'r') as f:
        for line in f:
            result = json.loads(line.strip())
            run_id = tuple(result[0])
            id_to_results[run_id].append({
                'loss': result[3]['loss'],
                'epochs': result[1]
            })

    # Match results to config
    with open(os.path.join(results_dir, 'configs.json'), 'r') as f:
        for line in f:
            config = json.loads(line.strip())
            run_id = tuple(config[0])
            if run_id not in id_to_results:
                print('Id "{}" in configs but not in results, skipping.'.format(run_id))
                continue
            for result in id_to_results[run_id]:
                result.update(**config[1])

    # Convert to list sorted by loss
    results = [result for result_list in id_to_results.values() for result in result_list]
    results.sort(key=lambda result: result['loss'])

    return results


class MPNWorker(Worker):
    def __init__(self, args: Namespace, **kwargs):
        super(MPNWorker, self).__init__(**kwargs)
        self.args = args
        self.sign = -1 if args.metric in ['auc', 'pr-auc', 'r2'] else 1

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
        args.epochs = math.ceil(budget)

        # Run training
        scores = run_training(args)

        return {'loss': self.sign * np.mean(scores)}  # BOHB optimizer tries to minimize loss

    @staticmethod
    def get_configspace():
        """
        Build configuration space.

        :return: A ConfigSpace object containing the configurations to try.
        """
        cs = CS.ConfigurationSpace()

        cs.add_hyperparameters([
            CSH.UniformIntegerHyperparameter('hidden_size', lower=150, upper=1800),
            CSH.UniformIntegerHyperparameter('depth', lower=2, upper=9),
            # TODO: allow virtual edges - currently breaks featurization b/c memoizes different feature vector sizes
            # CSH.CategoricalHyperparameter('virtual_edges', choices=[True, False])
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
    bohb.run(n_iterations=args.n_iterations)
    bohb.shutdown(shutdown_workers=True)
    NS.shutdown()


if __name__ == '__main__':
    args = parse_hyper_opt_args()
    optimize_hyperparameters(args)
