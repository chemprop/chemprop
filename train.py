"""Trains a chemprop model on a dataset."""

from warnings import warn

warn(
    "You have installed a development release of Chemprop (1.6.1.dev0) which is not "
    "guaranteed to be ready for production. It it intended only for testing, new features "
    "like SSL training and DDP.",
    UserWarning,
)

from chemprop.train import chemprop_train


if __name__ == '__main__':
    chemprop_train()
