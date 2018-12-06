from argparse import ArgumentParser
import os
import torch
from chemprop.run.parsing import parse_predict_args
from chemprop.run.predict import make_predictions

if __name__ == '__main__':
    args = parse_predict_args()
    make_predictions(args)
