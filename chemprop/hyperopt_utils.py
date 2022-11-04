from chemprop.args import HyperoptArgs
import os
import pickle
from typing import List, Dict
import csv
import json

from hyperopt import Trials

from chemprop.constants import HYPEROPT_SEED_FILE_NAME
from chemprop.utils import makedirs

def merge_trials(trials: Trials, new_trials_data: List[Dict]) -> Trials:
    """
    Merge a hyperopt trials object with the contents of another hyperopt trials object.

    :param trials: A hyperopt trials object containing trials data, organized into hierarchical dictionaries.
    :param trials_data: The contents of a hyperopt trials object, `Trials.trials`.
    :return: A hyperopt trials object, merged from the two inputs.
    """
    max_tid = 0
    if len(trials.trials) > 0:
        max_tid = max([trial['tid'] for trial in trials.trials])

    for trial in new_trials_data:
        tid = trial['tid'] + max_tid + 1 #trial id needs to be unique among this list of ids.
        hyperopt_trial = Trials().new_trial_docs(
                tids=[None],
                specs=[None],
                results=[None],
                miscs=[None])
        hyperopt_trial[0] = trial
        hyperopt_trial[0]['tid'] = tid
        hyperopt_trial[0]['misc']['tid'] = tid
        for key in hyperopt_trial[0]['misc']['idxs'].keys():
            hyperopt_trial[0]['misc']['idxs'][key] = [tid]
        trials.insert_trial_docs(hyperopt_trial) 
        trials.refresh()
    return trials


def load_trials(dir_path: str, previous_trials: Trials = None) -> Trials:
    """
    Load in trials from each pickle file in the hyperopt checkpoint directory.
    Checkpoints are newly loaded in at each iteration to allow for parallel entries
    into the checkpoint folder by independent hyperoptimization instances.

    :param dir_path: Path to the directory containing hyperopt checkpoint files.
    :param previous_trials: Any previously generated trials objects that the loaded trials will be merged with.
    :return: A trials object containing the merged trials from all checkpoint files.
    """

    # List out all the pickle files in the hyperopt checkpoint directory
    hyperopt_checkpoint_files = [os.path.join(dir_path, path) for path in os.listdir(dir_path) if '.pkl' in path]

    # Load hyperopt trials object from each file
    loaded_trials = Trials()
    if previous_trials is not None:
        loaded_trials = merge_trials(loaded_trials, previous_trials.trials)

    for path in hyperopt_checkpoint_files:
        with open(path,'rb') as f:
            trial = pickle.load(f)
            loaded_trials = merge_trials(loaded_trials, trial.trials)
    
    return loaded_trials


def save_trials(dir_path: str, trials: Trials, hyperopt_seed: int) -> None:
    """
    Saves hyperopt trial data as a `.pkl` file.

    :param dir_path: Path to the directory containing hyperopt checkpoint files.
    :param trials: A trials object containing information on a completed hyperopt iteration.
    """
    new_fname = f'{hyperopt_seed}.pkl'
    existing_files = os.listdir(dir_path)
    if new_fname in existing_files:
        raise ValueError(f'When saving trial with unique seed {hyperopt_seed}, found that a trial with this seed already exists.')
    pickle.dump(trials, open(os.path.join(dir_path, new_fname), 'wb'))


def get_hyperopt_seed(seed: int, dir_path: str) -> int:
    """
    Assigns a seed for hyperopt calculations. Each iteration will start with a different seed.

    :param seed: The initial attempted hyperopt seed.
    :param dir_path: Path to the directory containing hyperopt checkpoint files.
    :return: An integer for use as hyperopt random seed.
    """

    seed_path = os.path.join(dir_path,HYPEROPT_SEED_FILE_NAME)
    
    seeds = []
    if os.path.exists(seed_path):
        with open(seed_path, 'r') as f:
            seed_line = next(f)
            seeds.extend(seed_line.split())
    else:
        makedirs(seed_path, isfile=True)

    seeds = [int(sd) for sd in seeds]

    while seed in seeds:
        seed += 1
    seeds.append(seed)

    write_line = " ".join(map(str, seeds)) + '\n'

    with open(seed_path, 'w') as f:
        f.write(write_line)
    
    return seed


def load_manual_trials(manual_trials_dirs: List[str], param_keys: List[str], hyperopt_args: HyperoptArgs) -> Trials:
    """
    Function for loading in manual training runs as trials for inclusion in hyperparameter search.
    Trials must be consistent in all arguments with trials that would be generated in hyperparameter optimization.

    :param manual_trials_dirs: A list of paths to save directories for the manual trials, as would include test_scores.csv and args.json.
    :param param_keys: A list of the parameters included in the hyperparameter optimization.
    :param hyperopt_args: The arguments for the hyperparameter optimization job.
    :return: A hyperopt trials object including all the loaded manual trials.
    """
    matching_args = [ # manual trials must occupy the same space as the hyperparameter optimization search. This is a non-extensive list of arguments to check to see if they are consistent.
        'number_of_molecules',
        'aggregation',
        'num_folds',
        'ensemble_size',
        'max_lr',
        'init_lr',
        'final_lr',
        'activation',
        'metric',
        'bias',
        'epochs',
        'explicit_h',
        'reaction',
        'split_type',
        'warmup_epochs',
    ]

    manual_trials_data = []
    for i, trial_dir in enumerate(manual_trials_dirs):

        # Extract trial data from test_scores.csv
        with open(os.path.join(trial_dir, 'test_scores.csv')) as f:
            reader=csv.reader(f)
            next(reader)
            read_line=next(reader)
        mean_score = float(read_line[1])
        std_score = float(read_line[2])
        loss = (1 if hyperopt_args.minimize_score else -1) * mean_score

        # Extract argument data from args.json
        with open(os.path.join(trial_dir, 'args.json')) as f:
            trial_args = json.load(f)

        # Check for differences in manual trials and hyperopt space
        if 'hidden_size' in param_keys:
            if trial_args['hidden_size'] != trial_args['ffn_hidden_size']:
                raise ValueError(f'The manual trial in {trial_dir} has a hidden_size {trial_args["hidden_size"]} '
                f'that does not match its ffn_hidden_size {trial_args["ffn_hidden_size"]}, as it would in hyperparameter search.')
        for arg in matching_args:
            if arg not in param_keys:
                if getattr(hyperopt_args,arg) != trial_args[arg]:
                    raise ValueError(f'Manual trial {trial_dir} has different training argument {arg} than the hyperparameter optimization search trials.')

        # Construct data dict
        param_dict = {key: trial_args[key] for key in param_keys}
        vals_dict = {key: [param_dict[key]] for key in param_keys}
        idxs_dict = {key: [i] for key in param_keys}
        results_dict = {
            'loss': loss,
            'status': 'ok',
            'mean_score': mean_score,
            'std_score': std_score,
            'hyperparams': param_dict,
            'num_params': 0,
        }
        misc_dict = {
            'tid': i,
            'cmd': ('domain_attachment', 'FMinIter_Domain'),
            'workdir': None,
            'idxs': idxs_dict,
            'vals': vals_dict,
        }
        trial_data = {
            'state': 2,
            'tid': i,
            'spec': None,
            'result': results_dict,
            'misc': misc_dict,
            'exp_key': None,
            'owner': None,
            'version': 0,
            'book_time': None,
            'refresh_time': None,
        }
        manual_trials_data.append(trial_data)

    trials = Trials()
    trials = merge_trials(trials=trials, new_trials_data=manual_trials_data)
    return trials
