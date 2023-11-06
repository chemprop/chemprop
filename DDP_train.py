"""Trains a chemprop model on a dataset with DDP."""

from chemprop.train import chemprop_DDP_train


if __name__ == '__main__':
    chemprop_DDP_train()
