"""Loads a trained chemprop model checkpoint and encode latent fingerprint vectors for the molecules in a dataset.
    Uses the same command line arguments as predict."""

from chemprop.train import chemprop_fingerprint

if __name__ == '__main__':
    chemprop_fingerprint()
