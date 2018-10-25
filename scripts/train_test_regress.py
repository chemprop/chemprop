import os
print(os.getcwd())
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

from optparse import OptionParser
# ensure .../chemprop is on PYTHONPATH
from chemprop.mpn import *
from chemprop.dio import load_split_data
from chemprop.run import load_chemprop_model_architecture
from chemprop.run import set_processor


parser = OptionParser()
parser.add_option("-t", "--data_train", dest="data_train_path")
parser.add_option("-q", "--data_test", dest="data_test_path")
parser.add_option("-v", "--valid_split", dest="valid_split", default=.25)
parser.add_option("-z", "--test_split", dest="test_split", default=.0)
parser.add_option("-m", "--save_dir", dest="save_path")
parser.add_option("-b", "--batch", dest="batch_size", default=50)
parser.add_option("-w", "--hidden", dest="hidden_size", default=300)
parser.add_option("-d", "--depth", dest="depth", default=3)
parser.add_option("-e", "--epoch", dest="epoch", default=30)
parser.add_option("-p", "--dropout", dest="dropout", default=0)
parser.add_option("-x", "--metric", dest="metric", default='rmse')
parser.add_option("-s", "--seed", dest="seed", default=1)
parser.add_option("-c", "--scale", dest="scale", default=True)

opts, args = parser.parse_args()

batch_size = int(opts.batch_size)
depth = int(opts.depth)
hidden_size = int(opts.hidden_size)
num_epoch = int(opts.epoch)
dropout = float(opts.dropout)

if not os.path.isdir(opts.save_path):
    os.makedirs(opts.save_path)

train, valid, _ = load_split_data(opts.data_train_path, opts.valid_split, opts.test_split, opts.seed, opts.scale) # test data in separate file => split=0
test, _, _      = load_split_data(opts.data_test_path, 0., 0., opts.seed, opts.scale) #abuse function to load test data
print(len(train), len(valid), len(test))
num_tasks = len(train[0][1])
print("Number of tasks:", num_tasks)

model = load_chemprop_model_architecture(num_tasks=num_tasks)
model, loss_fn = set_processor(model)

optimizer = optim.Adam(model.parameters(), lr=1e-3)
scheduler = lr_scheduler.ExponentialLR(optimizer, 0.9)
scheduler.step()

from chemprop.run import train_model, valid_loss
best_model_wts = train_model(model, train, valid,
                             num_epoch, batch_size,
                             optimizer, scheduler, loss_fn,
                             opts.metric,
                             opts.save_path)
model.load_state_dict(torch.load(opts.save_path + "/model.best"))
print("test error: %.4f" % valid_loss(model, test, opts.metric))
