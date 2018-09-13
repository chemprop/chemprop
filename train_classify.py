import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
import os

import math, random, sys
from optparse import OptionParser
from collections import deque

from mpn import *

parser = OptionParser()
parser.add_option("-t", "--train", dest="train_path")
parser.add_option("-v", "--valid", dest="valid_path")
parser.add_option("-z", "--test", dest="test_path")
parser.add_option("-m", "--save_dir", dest="save_path")
parser.add_option("-b", "--batch", dest="batch_size", default=50)
parser.add_option("-w", "--hidden", dest="hidden_size", default=300)
parser.add_option("-d", "--depth", dest="depth", default=3)
parser.add_option("-e", "--epoch", dest="epoch", default=30)
parser.add_option("-p", "--dropout", dest="dropout", default=0)
parser.add_option("-c", "--metric", dest="metric", default='roc')
parser.add_option("-a", "--anneal", dest="anneal", default=-1)
opts,args = parser.parse_args()
   
batch_size = int(opts.batch_size)
depth = int(opts.depth)
hidden_size = int(opts.hidden_size)
num_epoch = int(opts.epoch)
dropout = float(opts.dropout)
anneal_iter = int(opts.anneal)

if not os.path.isdir(opts.save_path):
    raise ValueError('save directory does not exist')

def get_data(path):
    data = []
    func = lambda x : int(float(x)) if x != '' else -1
    with open(path) as f:
        f.readline()
        for line in f:
            vals = line.strip("\r\n ").split(',')
            smiles = vals[0]
            vals = [func(x) for x in vals[1:]]
            data.append((smiles,vals))
    return data

train = get_data(opts.train_path)
valid = get_data(opts.valid_path)
test = get_data(opts.test_path)

num_tasks = len(train[0][1])
print "Number of tasks:", num_tasks

encoder = MPN(hidden_size, depth, dropout=dropout)
model = nn.Sequential(
        encoder,
        nn.Linear(hidden_size, hidden_size), 
        nn.ReLU(), 
        nn.Linear(hidden_size, num_tasks * 2)
    )
#class_weight = torch.Tensor([1,5])
#loss_fn = nn.CrossEntropyLoss(weight=class_weight, ignore_index=-1).cuda()
loss_fn = nn.CrossEntropyLoss(ignore_index=-1).cuda()
model = model.cuda()

for param in model.parameters():
    if param.dim() == 1:
        nn.init.constant(param, 0)
    else:
        nn.init.xavier_normal(param)

optimizer = optim.Adam(model.parameters(), lr=1e-3)
scheduler = lr_scheduler.ExponentialLR(optimizer, 0.9)
scheduler.step()

def valid_loss(data):
    model.train(False)
    all_preds = [[] for i in xrange(num_tasks)]
    all_labels = [[] for i in xrange(num_tasks)]

    for k in xrange(0, len(data), batch_size):
        batch = data[k:k+batch_size]
        mol_batch, label_batch = zip(*batch)
        mol_batch = mol2graph(mol_batch)

        preds = F.softmax(model(mol_batch).view(-1,num_tasks,2), dim=2)
        for i in xrange(num_tasks):
            for j in xrange(len(batch)):
                if label_batch[j][i] >= 0:
                    all_preds[i].append(preds.data[j][i][1])
                    all_labels[i].append(label_batch[j][i])

    model.train(True)

    #compute roc-auc
    if opts.metric == 'roc':
        res = []
        for i in xrange(num_tasks):
            if sum(all_labels[i]) == 0: continue
            if min(all_labels[i]) == 1: continue
            res.append( roc_auc_score(all_labels[i], all_preds[i]) )
        return sum(res) / len(res)

    #compute prc-auc
    res = 0
    for i in xrange(num_tasks):
        r,p,t = precision_recall_curve(all_labels[i], all_preds[i])
        val = auc(p,r)
        if math.isnan(val): val = 0
        res += val
    return res / num_tasks

best_loss = 0
zero = create_var(torch.zeros(1))
for epoch in xrange(num_epoch):
    mse,it = 0,0
    print "learning rate: %.6f" % scheduler.get_lr()[0]
    for i in xrange(0, len(train), batch_size):
        batch = train[i:i+batch_size]
        mol_batch, label_batch = zip(*batch)
        mol_batch = mol2graph(mol_batch)
        labels = create_var(torch.LongTensor(label_batch))

        model.zero_grad()
        preds = model(mol_batch)
        loss = loss_fn(preds.view(-1,2), labels.view(-1))
        mse += loss
        it += batch_size
        loss.backward()
        optimizer.step()

        if i % 1000 == 0:
            pnorm = math.sqrt(sum([p.norm().data[0] ** 2 for p in model.parameters()]))
            gnorm = math.sqrt(sum([p.grad.norm().data[0] ** 2 for p in model.parameters()]))
            print "loss=%.4f,PNorm=%.2f,GNorm=%.2f" % (mse / it, pnorm, gnorm)
            sys.stdout.flush()
            mse,it = 0,0

        if anneal_iter > 0 and i % anneal_iter == 0:
            scheduler.step()
            torch.save(model.state_dict(), opts.save_path + "/model.iter-%d-%d" % (epoch,i))

    if anneal_iter == -1: #anneal == len(train)
        scheduler.step()

    cur_loss = valid_loss(valid)
    print "validation error: %.4f" % cur_loss
    if opts.save_path is not None:
        torch.save(model.state_dict(), opts.save_path + "/model.iter-" + str(epoch))
        if cur_loss > best_loss:
            best_loss = cur_loss
            torch.save(model.state_dict(), opts.save_path + "/model.best")

model.load_state_dict(torch.load(opts.save_path + "/model.best"))
print "test error: %.4f" % valid_loss(test)
