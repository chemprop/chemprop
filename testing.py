import csv

import numpy as np
import pytorch_lightning as pl
from sklearn.preprocessing import StandardScaler

from chemprop.v2 import data
from chemprop.v2 import featurizers
from chemprop.v2.models import modules, models
import torch

with open("/home/gridsan/nmorgan/MW.csv") as fid:
    reader = csv.reader(fid)
    next(reader)
    smis, scores = zip(*[(smi, float(score)) for smi, score in reader])
scores = np.array(scores).reshape(-1,1)
all_data = [data.MoleculeDatapoint(smi, target) for smi, target in zip(smis, scores)]

generator = torch.Generator().manual_seed(12)
train_data, val_data, test_data = torch.utils.data.random_split(all_data, [0.81,0.09,0.1], generator=generator)

featurizer = featurizers.MoleculeFeaturizer()

train_dset = data.MoleculeDataset(train_data, featurizer)
val_dset = data.MoleculeDataset(val_data, featurizer)
test_dset = data.MoleculeDataset(test_data, featurizer)

scaler = StandardScaler().fit(np.array([d._targets for d in train_dset.data]))

molenc = modules.molecule_block()
mpnn = models.RegressionMPNN(molenc, 1, scaler=scaler)

train_loader = data.MolGraphDataLoader(train_dset, num_workers=4)
val_loader = data.MolGraphDataLoader(val_dset, num_workers=4, shuffle=False)
test_loader = data.MolGraphDataLoader(test_dset, num_workers=4, shuffle=False)

trainer = pl.Trainer(
    enable_progress_bar=True,
    devices=1,
    max_epochs=1000,
    accelerator="gpu",
    default_root_dir="/home/gridsan/nmorgan/chemprop/nathan/"
)

trainer.fit(mpnn, train_loader, val_loader)

with open("/home/gridsan/nmorgan/MW_smi.csv") as fid:
    reader = csv.reader(fid)
    next(reader)
    smis = [smi[0] for smi in reader]

pred_smis = [data.MoleculeDatapoint(smi) for smi in smis]

pred_dset = data.MoleculeDataset(pred_smis, featurizer)
pred_loader = data.MolGraphDataLoader(pred_dset, num_workers=4, shuffle=False)
trainer.predict(mpnn, pred_loader)