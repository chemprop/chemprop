"""Loads a previously trained chemprop model and estimates the uncertainty in each prediction. 
    Optionally also calibrate the uncertainty metrics using a separate validation set."""

from chemprop.uncertainty import chemprop_uncertainty

if __name__ == '__main__':
    chemprop_uncertainty()
