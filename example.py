import csv
import sys

from lightning import pytorch as pl
import numpy as np
from sklearn.model_selection import train_test_split

from chemprop.v2 import data
from chemprop.v2 import featurizers
from chemprop.v2.models import loss, modules, models, metrics


featurizer = featurizers.MoleculeFeaturizer()
mp = modules.BondMessageBlock()
agg = modules.MeanAggregation()
ffn = modules.RegressionFFN()
mpnn = models.MPNN(mp, agg, ffn, [metrics.RMSEMetric()])

print(mpnn)

with open(sys.argv[1]) as fid:
    reader = csv.reader(fid)
    next(reader)
    smis, scores = zip(*[(smi, float(score)) for smi, score in reader])
scores = np.array(scores).reshape(-1, 1)
all_data = [data.MoleculeDatapoint(smi, target) for smi, target in zip(smis, scores)]

train_data, val_test_data = train_test_split(all_data, test_size=0.1)
val_data, test_data = train_test_split(val_test_data, test_size=0.5)

train_dset = data.MoleculeDataset(train_data, featurizer)
scaler = train_dset.normalize_targets()

val_dset = data.MoleculeDataset(val_data, featurizer)
val_dset.normalize_targets(scaler)
test_dset = data.MoleculeDataset(test_data, featurizer)
test_dset.normalize_targets(scaler)

train_loader = data.MolGraphDataLoader(train_dset, num_workers=0)
val_loader = data.MolGraphDataLoader(val_dset, num_workers=0, shuffle=False)
test_loader = data.MolGraphDataLoader(test_dset, num_workers=0, shuffle=False)

trainer = pl.Trainer(
    logger=False,
    enable_checkpointing=False,
    enable_progress_bar=True,
    accelerator="auto",
    devices=1,
    max_epochs=20,
)
trainer.fit(mpnn, train_loader, val_loader)
results = trainer.test(mpnn, test_loader)
print(results)
