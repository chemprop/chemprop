import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc

import math, random, sys
from optparse import OptionParser
from collections import deque

from mpn import *

parser = OptionParser()
parser.add_option("-t", "--test", dest="test_path")
parser.add_option("-m", "--model", dest="model_path")
parser.add_option("-b", "--batch", dest="batch_size", default=50)
parser.add_option("-w", "--hidden", dest="hidden_size", default=300)
parser.add_option("-d", "--depth", dest="depth", default=3)
parser.add_option("-p", "--dropout", dest="dropout", default=0)
parser.add_option("-n", "--num_task", dest="num_task", default=1)
opts,args = parser.parse_args()
   
batch_size = int(opts.batch_size)
depth = int(opts.depth)
hidden_size = int(opts.hidden_size)
dropout = float(opts.dropout)
num_tasks = int(opts.num_task)

def get_data(path):
    data = []
    with open(path) as f:
        f.readline()
        for line in f:
            smiles = line.strip("\r\n ").split(',')[0]
            data.append(smiles)
    return data

test = get_data(opts.test_path)

encoder = MPN(hidden_size, depth, dropout=dropout)
model = nn.Sequential(
        encoder,
        nn.Linear(hidden_size, hidden_size), 
        nn.ReLU(), 
        nn.Linear(hidden_size, num_tasks)
    )
model.load_state_dict(torch.load(opts.model_path))
model = model.cuda()

model.train(False)
for k in xrange(0, len(test), batch_size):
    batch = test[k:k+batch_size]
    mol_batch = mol2graph(batch)
    score = model(mol_batch)
    for i in xrange(len(batch)):
        for j in xrange(num_tasks):
            print "%.4f" % score[i,j],
        print
