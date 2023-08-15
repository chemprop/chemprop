import csv

from lightning import pytorch as pl
import numpy as np
from sklearn.model_selection import train_test_split

from chemprop.v2 import data
from chemprop.v2 import featurizers
from chemprop.v2.models import modules, models, metrics


featurizer = featurizers.MoleculeFeaturizer()
mp = modules.BondMessageBlock(*featurizer.shape)
agg = modules.MeanAggregation()
ffn = modules.RegressionFFN(mp.output_dim, 1)
mpnn = models.MPNN(mp, agg, ffn, [metrics.MSEMetric()])
print(mpnn)

with open("./data/freesolv.csv") as fid:
    reader = csv.reader(fid)
    next(reader)
    smis, scores = zip(*[(smi, float(score)) for smi, score in reader])
scores = np.array(scores).reshape(-1, 1)
all_data = [data.MoleculeDatapoint(smi, target) for smi, target in zip(smis, scores)]

train_data, val_test_data = train_test_split(all_data, test_size=0.1)
val_data, test_data = train_test_split(val_test_data, test_size=0.5)

train_dset = data.MoleculeDataset(train_data, featurizer)
scaler = train_dset.normalize_targets()

import pdb; pdb.set_trace()

val_dset = data.MoleculeDataset(val_data, featurizer)
val_dset.normalize_targets(scaler)
test_dset = data.MoleculeDataset(test_data, featurizer)
test_dset.normalize_targets(scaler)

train_loader = data.MolGraphDataLoader(train_dset, num_workers=4)
val_loader = data.MolGraphDataLoader(val_dset, num_workers=4, shuffle=False)
test_loader = data.MolGraphDataLoader(test_dset, num_workers=4, shuffle=False)

trainer = pl.Trainer(
    logger=False,
    enable_checkpointing=False,
    enable_progress_bar=True,
    accelerator="gpu",
    devices=1,
    max_epochs=10,
)
trainer.fit(mpnn, train_loader, val_loader)
results = trainer.test(mpnn, test_loader)
print(results)
