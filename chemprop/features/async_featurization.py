from argparse import Namespace
from typing import List, Tuple, Union

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

import torch
torch.multiprocessing.set_sharing_strategy('file_system')
from torch.multiprocessing import Queue, Pool

from chemprop.data import MoleculeDataset
from .featurization import mol2graph

def mol2graph_helper(pair):
    batch, args = pair
    return batch, mol2graph(batch.smiles(), args)

def async_mol2graph(q: Queue, 
                    data: MoleculeDataset, 
                    args: Namespace,
                    num_iters: int,
                    iter_size: int,
                    exit_q: Queue,
                    last_batch: bool=False):
    batches = []
    for i in range(0, num_iters, iter_size):  # will only go up to max size of queue, then yield
        if not last_batch and i + args.batch_size > len(data):
            break
        batch = MoleculeDataset(data[i:i + args.batch_size])
        batches.append(batch)
        if len(batches) == args.batches_per_queue_group:  # many at a time, since synchronization is expensive
            with Pool() as pool:
                processed_batches = pool.map(mol2graph_helper, [(batch, args) for batch in batches])
            q.put(processed_batches)
            batches = []
    if len(batches) > 0:
        with Pool() as pool:
            processed_batches = pool.map(mol2graph_helper, [(batch, args) for batch in batches])
        q.put(processed_batches)
    exit_q.get()  # prevent from exiting until main process tells it to; otherwise we apparently can't read the end of the queue and crash