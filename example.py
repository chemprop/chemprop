import csv

import numpy as np
import pytorch_lightning as pl
from sklearn.model_selection import train_test_split

from chemprop.v2 import data
from chemprop.v2 import featurizers
from chemprop.v2.models import modules, models


featurizer = featurizers.MoleculeFeaturizer()
molenc = modules.molecule_block()
mpnn = models.RegressionMPNN(molenc, 1)
print(mpnn)

with open("tests/data/regression.csv") as fid:
    reader = csv.reader(fid)
    next(reader)
    smis, scores = zip(*[(smi, float(score)) for smi, score in reader])
scores = np.array(scores).reshape(-1, 1)
all_data = [data.MoleculeDatapoint(smi, target) for smi, target in zip(smis, scores)]

train_val_data, test_data = train_test_split(all_data, test_size=0.1)
train_data, val_data = train_test_split(train_val_data, test_size=0.1)

train_dset = data.MoleculeDataset(train_data, featurizer)
scaler = train_dset.normalize_targets()

val_dset = data.MoleculeDataset(val_data, featurizer)
val_dset.normalize_targets(scaler)
test_dset = data.MoleculeDataset(test_data, featurizer)
test_dset.normalize_targets(scaler)

train_loader = data.MolGraphDataLoader(train_dset, num_workers=4)
val_loader = data.MolGraphDataLoader(val_dset, num_workers=4, shuffle=False)
test_loader = data.MolGraphDataLoader(test_dset, num_workers=4, shuffle=False)

trainer = pl.Trainer(
    # logger=False,
    enable_checkpointing=False,
    enable_progress_bar=True,
    accelerator="gpu",
    devices=1,
    max_epochs=5,
)
trainer.fit(mpnn, train_loader, val_loader)
results = trainer.test(mpnn, test_loader)
print(results)
