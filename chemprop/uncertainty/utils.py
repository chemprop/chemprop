#!/usr/bin/python3

import numpy as np
from scipy.special import erfinv

def fraction_below_value(data: np.ndarray, value: float):
    """
    
    """
    assert data.ndim == 2 # shape(data,tasks)
    data_size = data.shape[0]
    task_size = data.shape[1]
    max_value = np.max(data, axis=0)
    min_value = np.min(data, axis=0)
    virtual_lower = np.sum(data < value, axis=0) - 1 #shape(tasks)
    virtual_upper = virtual_lower + 1

    virtual_lower[min_value >= value] = 0
    virtual_upper[min_value >= value] = 0
    virtual_lower[max_value <= value] = data_size - 1
    virtual_upper[max_value <= value] = data_size - 1

    sorted_array = np.sort(data, axis=0)
    value_lower = sorted_array[virtual_lower, np.arange(task_size)]
    value_upper = sorted_array[virtual_upper, np.arange(task_size)]
    interpolation_weight = (value - value_lower) / (value_upper - value_lower)
    interpolation_index = interpolation_weight * (virtual_upper - virtual_lower) + virtual_lower
    frac = (interpolation_index) / (data_size-1)
    return frac # shape(tasks)

def calibration_normal_auc(abs_z_scores: np.ndarray, num_bins: int = 101):
    """
    
    """
    abs_z = np.abs(abs_z_scores) # shape(data,tasks)
    bins = np.linspace(0,1,num_bins)
    zbins = erfinv(bins) * np.sqrt(2) # first value is 0 and last is inf
    fracs = np.zeros((zbins.shape[0],abs_z.shape[1])) # shape(bins, tasks)
    for i, zbin in enumerate(zbins):
        fraction = fraction_below_value(abs_z, zbin)
        fracs[i] = fraction
    auc = np.sum(fracs - np.expand_dims(bins, 1), axis=0) # Not scaled to step size, not halving endpoints because they are both zero.
    return auc
