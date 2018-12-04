from .morgan_fingerprint import morgan_fingerprint
from functools import partial

#TODO(kernel) implement more kernel functions, e.g. rdkit?

def get_kernel_func(kernel_func_name):
    if kernel_func_name == 'morgan':
        return partial(morgan, use_counts=False)
    
    if kernel_func_name == 'morgan_count':
        return partial(morgan, use_counts=True)

    raise ValueError('kernel function "{}" not supported.'.format(kernel_func_name))

def morgan(datapoint1, datapoint2, args, use_counts=False):
    fp1 = morgan_fingerprint(datapoint1.mol, use_counts=use_counts)
    fp2 = morgan_fingerprint(datapoint2.mol, use_counts=use_counts)
    return [(fp1 * fp2).sum()]