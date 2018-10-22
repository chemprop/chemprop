from chemprop.mpn import *
import torch
import torch.nn as nn
import math, sys
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import copy
from chemprop.utils import cuda_available

def set_processor(model):
    print("Cuda available: ", cuda_available())
    if cuda_available():
        loss_fn = nn.MSELoss(reduce=False).cuda()
        model = model.cuda()
    else:
        loss_fn = nn.MSELoss(reduce=False).cpu()
        model = model.cpu()
    return model, loss_fn


def load_chemprop_model_architecture(hidden_size=300, depth=3, num_tasks=1):
    encoder = MPN(hidden_size, depth)
    model = nn.Sequential(
        encoder,
        nn.Linear(hidden_size, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, num_tasks)
    )
    for param in model.parameters():
        if param.dim() == 1:
            nn.init.constant(param, 0)
        else:
            nn.init.xavier_normal(param)
    return model



# Update with stored weights
def load_stored_weights(model, fn):
    _model = torch.load(fn)
    model.load_state_dict(_model)
    return model


def set_model_retrain(model, hidden_size, num_tasks, allow_parameters=True):
    if allow_parameters:
        model[3] = nn.Linear(hidden_size, num_tasks)
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
    else:
        for param in model.parameters(): # Freeze  parameter layers so they are not retrained
            param.requires_grad = False
        model[3] = nn.Linear(hidden_size, num_tasks)
        optimizer = optim.Adam(model[3].parameters(), lr=1e-3)
    scheduler = lr_scheduler.ExponentialLR(optimizer, 0.9)
    return model, optimizer, scheduler


def get_batch(i, data, batch_size=300):
    batch = data[i:i+batch_size]
    mol_batch, label_batch = zip(*batch)
    return mol_batch, label_batch


def get_predictions(model, data, num_tasks=1, batch_size=300):
    val=np.zeros((len(data), num_tasks))
    for i in xrange(0, len(data), batch_size):
        mol_batch, label_batch = get_batch(i, data, batch_size)
        mol_batch = mol2graph(mol_batch)
        val[i:i+batch_size] = model(mol_batch).detach().numpy()
    return val


def valid_loss(model, data, metric, batch_size=100):
    num_tasks = len(data[0][1])
    if metric == 'mae':
        val_loss = nn.L1Loss(reduce=False)
    else:
        val_loss = nn.MSELoss(reduce=False)

    err = torch.zeros(num_tasks)
    ndata = torch.zeros(num_tasks)
    model.train(False)
    for i in xrange(0, len(data), batch_size):
        batch = data[i:i+batch_size]
        mol_batch, label_batch = zip(*batch)
        mol_batch = mol2graph(mol_batch)
        mask = [map(lambda x: x is not None, lb) for lb in label_batch]
        mask = create_var(torch.Tensor(mask))
        label_batch = [map(lambda x: 0 if x is None else x, lb) for lb in label_batch]
        labels = create_var(torch.Tensor(label_batch))
        preds = model(mol_batch)
        loss = val_loss(preds, labels) * mask
        ndata += mask.data.sum(dim=0).cpu()
        err += loss.data.sum(dim=0).cpu()
    model.train(True)
    err = err / ndata
    print(err, err.sqrt(), num_tasks)
    if metric == 'rmse':
        return err.sqrt().sum() / num_tasks
    else:
        return err.sum() / num_tasks


def train_model(model, train, valid, num_epoch, batch_size, optimizer, scheduler, loss_fn, metric, save_path,
                verbose=True, saveiter=False):
    import os
    if save_path is not None:
        if not os.path.exists(save_path):
            os.makedirs(save_path)
    num_tasks = len(train[0][1])
    best_loss = 1e5
    best_model_wts = []
    for epoch in xrange(num_epoch):
        mse, it = 0,0
        print("learning rate: %.6f" % scheduler.get_lr()[0])
        for i in xrange(0, len(train), batch_size):
            batch = train[i:i+batch_size]
            mol_batch, label_batch = zip(*batch)
            mol_batch = mol2graph(mol_batch)
            mask = [map(lambda x: x is not None, lb) for lb in label_batch]
            mask = create_var(torch.Tensor(mask))
            label_batch = [map(lambda x: 0 if x is None else x, lb) for lb in label_batch]
            labels = create_var(torch.Tensor(label_batch))

            model.zero_grad()
            preds = model(mol_batch)
            loss = loss_fn(preds, labels) * mask
            loss = loss.sum() / mask.sum()
            mse += loss.data[0] * batch_size
            loss = loss * num_tasks
            it += batch_size
            loss.backward()
            optimizer.step()

            if verbose & (i % 1000 == 0):
                pnorm = math.sqrt(sum([p.norm().data[0] ** 2 for p in model.parameters()]))
                #gnorm = math.sqrt(sum([p.grad.norm().data[0] ** 2 for p in model.parameters()]))
                #print("RMSE=%.4f,PNorm=%.2f,GNorm=%.2f" % (math.sqrt(mse / it), pnorm, gnorm))
                print("RMSE=%.4f,PNorm=%.2f" % (math.sqrt(mse / it), pnorm))
                sys.stdout.flush()
                mse, it = 0, 0

        scheduler.step()
        cur_loss = valid_loss(model, valid, metric)
        print("validation error: %.4f" % cur_loss)
        if saveiter:
            if save_path is not None:
                torch.save(model.state_dict(), save_path + "/model.iter-" + str(epoch))
        if cur_loss < best_loss:
            best_loss = cur_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            if save_path is not None:
                torch.save(model.state_dict(), save_path + "/model.best")

    return best_model_wts