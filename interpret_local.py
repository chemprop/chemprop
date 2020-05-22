"""Loads a trained model checkpoint and makes predictions on a dataset."""

from chemprop.args import LocalInterpretArgs
from chemprop.train import make_predictions

if __name__ == '__main__':
    args = LocalInterpretArgs().parse_args()
    make_predictions(args, bayes_ensemble_grad=True)
