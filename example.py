import pdb
import numpy as np
import torch

from chemprop.data import v2 as data
from chemprop.featurizers import v2 as featurizers
from chemprop.models.v2 import encoders, models
from chemprop.models.v2.ptl import loss


featurizer = featurizers.MoleculeFeaturizer()
molenc = encoders.molecule_encoder()
rxnenc = encoders.reaction_encoder(1)
mpnn = models.MPNN(molenc, 1)
criterion = loss.MSELoss()

smis = ["c1ccccc1", "CCCC", "CC(=O)C"]
targets = np.random.rand(len(smis), 1)
datapoints = [data.MoleculeDatapoint(smi, target) for smi, target in zip(smis, targets)]
dset = data.MoleculeDataset(datapoints, featurizer)
train_loader = data.MolGraphDataLoader(dset)

target_weights = torch.ones(targets.shape[1])
optim = torch.optim.Adam(mpnn.parameters(), 3e-4)

mpnn.train()
for batch in train_loader:
    optim.zero_grad()

    bmg, X_vd, X_f, targets, weights, lt_targets, gt_targets = batch
    mask = torch.isfinite(targets)

    preds = mpnn((bmg, X_vd), X_f)
    pdb.set_trace()
    L = criterion(
        preds, targets, mask=mask, weights=weights, lt_targets=lt_targets, gt_targets=gt_targets
    )
    L = L * mask * target_weights

    l = L.sum() / mask.sum()

    l.backward()
    optim.step()

