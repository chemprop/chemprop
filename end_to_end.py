import os
import logging
import multiprocessing
import time

from parsing import get_parser, modify_args
from train import run_training, cross_validate
from hyper_opt import optimize_hyperparameters
from resplit_data import resplit
from avg_dups import average_duplicates
from predict import make_predictions
from utils import set_logger

logger = logging.getLogger('train')
logger.setLevel(logging.DEBUG)

def merge_train_val(args):
    with open(args.train_save, 'r') as tf, \
            open(args.val_save, 'r') as vf, \
            open(args.train_val_save, 'w') as tvf:
        for line in tf:
            tvf.write(line.strip())
            tvf.write('\n')
        vf.readline()
        for line in vf:
            tvf.write(line.strip())
            tvf.write('\n')

def reformat(text):
    return text.replace('null', 'None').replace('false', 'False').replace('true', 'True')

def prep_args_for_training(args):
    args.data_path = args.train_val_save
    # args.num_folds = 10
    # args.ensemble_size = 1
    # args.split_sizes = [0.8, 0.2, 0]
    best_config_index = None
    best_loss = float('inf')
    with open(os.path.join(args.results_dir, 'results.json'), 'r') as f:
        for line in f:
            result = eval(reformat(line.strip()))
            config_index, loss = result[0], result[3]['loss']
            if loss < best_loss:
                best_config_index, best_loss = config_index, loss
    with open(os.path.join(args.results_dir, 'configs.json'), 'r') as f:
        for line in f:
            result = eval(reformat(line.strip()))
            config_index = result[0]
            if config_index == best_config_index:
                config = result[1]
                args.depth = config['depth']
                args.hidden_size = config['hidden_size']
                if config['master_node']:
                    args.master_node = True
                    args.master_dim = config['master_dim']
                break

if __name__ == '__main__':
    parser = get_parser()
    parser.add_argument('--train_path', type=str, required=True,
                        help='Path to CSV file containing training data in chronological order')
    parser.add_argument('--val_path', type=str, required=True,
                        help='Path to CSV file containing val data in chronological order')
    parser.add_argument('--train_save', type=str, required=True,
                        help='Path to CSV file for new train data')
    parser.add_argument('--val_save', type=str, required=True,
                        help='Path to CSV file for new val data')
    parser.add_argument('--val_frac', type=float, default=0.2,
                        help='frac of data to use for validation')
    parser.add_argument('--train_val_save', type=str, required=True,
                        help='Path to CSV file for combined train and val data')
    
    parser.add_argument('--results_dir', type=str,
                        help='Path to directory where results will be saved')
    parser.add_argument('--port', type=int, default=9090,
                        help='Port for HpBandSter to use')
    parser.add_argument('--min_budget', type=int, default=5,
                        help='Minimum budget (number of iterations during training) to use')
    parser.add_argument('--max_budget', type=int, default=45,
                        help='Maximum budget (number of iterations during training) to use')
    parser.add_argument('--eta', type=int, default=2,
                        help='Factor by which to cut number of trials (1/eta trials remain)')
    parser.add_argument('--n_iterations', type=int, default=16,
                        help='Number of iterations of BOHB algorithm')
    parser.add_argument('--hyperopt_timeout', type=float, default=-1,
                        help='seconds to wait for hyperopt script')

    parser.add_argument('--test_path', type=str, required=True,
                        help='Path to CSV file containing testing data for which predictions will be made')
    parser.add_argument('--save_path', type=str, required=True,
                        help='Path to CSV file where predictions will be saved')
    parser.add_argument('--compound_names', action='store_true', default=False,
                        help='Save compound name in addition to predicted values (use when test file has compound names)')
    args = parser.parse_args()

    modify_args(args)

    resplit(args)
    merge_train_val(args)
    for path in [args.train_save, args.val_save, args.train_val_save]:
        args.data_path = path
        args.save_path = path
        average_duplicates(args)

    os.makedirs(args.results_dir, exist_ok=True)
    # if args.hyperopt_timeout > 0: #TODO for some reason this crashes
    #     start_time = time.time()
    #     p = multiprocessing.Process(target=optimize_hyperparameters, name="hyperopt", args=(args,))
    #     p.start()
    #     while p.is_alive():
    #         print(time.time() - start_time)
    #         if time.time() - start_time > args.hyperopt_timeout:
    #             p.terminate()
    #             p.join()
    #             break
    #         else:
    #             time.sleep(1)
    # else: #no timeout specified
    #    optimize_hyperparameters(args)
    optimize_hyperparameters(args) #TODO this seems to run forever?

    prep_args_for_training(args)
    set_logger(logger, args.save_dir, args.quiet)
    cross_validate(args)

    args.checkpoint_dir = args.save_dir
    args.no_cuda = False
    modify_args(args)
    make_predictions(args)



    