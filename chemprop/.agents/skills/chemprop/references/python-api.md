# Python API reference

Use the Python API when the CLI is not flexible enough: custom architectures, custom training
loops, or calling Chemprop from other code. The API is built on PyTorch Lightning. There is no
v1-style `chemprop.train` / `TrainArgs` API.

```python
from chemprop import data, featurizers, models, nn
```

Use attribute access (`nn.RegressionFFN`, `data.MoleculeDataset`) rather than `import *`.

## Train a regression model

```python
import pandas as pd
from lightning import pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint
from chemprop import data, featurizers, models, nn

df = pd.read_csv("data.csv")
smis = df["smiles"].values
ys = df[["target"]].values  # 2D: one row per molecule, one column per task; NaN = missing

all_data = [data.MoleculeDatapoint.from_smi(smi, y) for smi, y in zip(smis, ys)]
mols = [d.mol for d in all_data]
train_idx, val_idx, test_idx = data.make_split_indices(mols, "kmeans", (0.8, 0.1, 0.1))
# Both functions return one list per replicate, hence the [0] below.
train_data, val_data, test_data = data.split_data_by_indices(all_data, train_idx, val_idx, test_idx)

featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()
train_dset = data.MoleculeDataset(train_data[0], featurizer)
scaler = train_dset.normalize_targets()  # fit on train only
val_dset = data.MoleculeDataset(val_data[0], featurizer)
val_dset.normalize_targets(scaler)  # same scaler for validation
test_dset = data.MoleculeDataset(test_data[0], featurizer)  # do NOT normalize test data

train_loader = data.build_dataloader(train_dset, num_workers=0)
val_loader = data.build_dataloader(val_dset, num_workers=0, shuffle=False)
test_loader = data.build_dataloader(test_dset, num_workers=0, shuffle=False)

mp = nn.BondMessagePassing()  # d_h=300, depth=3 by default
agg = nn.MeanAggregation()
ffn = nn.RegressionFFN(
    input_dim=mp.output_dim,
    # predictions in original units
    output_transform=nn.UnscaleTransform.from_standard_scaler(scaler),
)
model = models.MPNN(mp, agg, ffn, batch_norm=True, metrics=[nn.metrics.RMSE(), nn.metrics.MAE()])

checkpointing = ModelCheckpoint(
    "checkpoints", "best-{epoch}-{val_loss:.2f}", "val_loss", mode="min"
)
trainer = pl.Trainer(
    logger=False, accelerator="auto", devices=1, max_epochs=50, callbacks=[checkpointing]
)
trainer.fit(model, train_loader, val_loader)

best = models.MPNN.load_from_checkpoint(checkpointing.best_model_path)
trainer.test(best, dataloaders=test_loader)  # logs test metrics
models.save_model("model.pt", best, output_columns=["target"])  # same format as CLI best.pt
```

Key points:

- `y` for each datapoint must be a 1D array with one entry per task (rows of `df[cols].values`).
- Target scaling only applies to regression. The `UnscaleTransform` is what turns predictions
  back into original units; without it, predictions are in standardized units.
- Pass the model to `trainer.test` as above. If you instead let Lightning load the best checkpoint
  (`trainer.test(dataloaders=...)`), Lightning 2.6+ needs `weights_only=False`, and older
  versions reject that argument.
- Always set `max_epochs`; the learning-rate schedule is computed from it.
- `val_loss` is the training loss on validation data and lower is better. The `metrics` are only
  logged (as `val/rmse`, ...); the loss comes from the predictor (`criterion=`).
- Split types for `make_split_indices`: `"random"`, `"scaffold_balanced"`,
  `"random_with_repeated_smiles"`, `"kennard_stone"`, `"kmeans"`.

## Predict with a saved model

```python
import numpy as np
import pandas as pd
import torch
from lightning import pytorch as pl
from chemprop import data, featurizers, models

model = models.MPNN.load_from_file("model.pt")  # .pt from save_model or CLI best.pt
# For a Lightning checkpoint use models.MPNN.load_from_checkpoint("path.ckpt")

df = pd.read_csv("new.csv")
dset = data.MoleculeDataset(
    [data.MoleculeDatapoint.from_smi(smi) for smi in df["smiles"]],
    featurizers.SimpleMoleculeMolGraphFeaturizer(),  # must match the featurizer used in training
)
loader = data.build_dataloader(dset, num_workers=0, shuffle=False)
with torch.inference_mode():
    trainer = pl.Trainer(logger=False, accelerator="auto", devices=1)
    preds = np.concatenate(trainer.predict(model, loader), axis=0)  # shape (n, n_tasks)
df["target_pred"] = preds[:, 0]
```

Mixture models load with `models.MulticomponentMPNN.load_from_file`, atom/bond models with
`models.MolAtomBondMPNN.load_from_file`.

## Other task types

Swap the predictor; everything else stays the same.

| Task | Predictor | Notes |
|---|---|---|
| regression | `nn.RegressionFFN(n_tasks=t)` | normalize targets + `UnscaleTransform` |
| regression with variance | `nn.MveFFN` | output `(n, t, 2)`: mean, variance |
| evidential regression | `nn.EvidentialFFN` | output `(n, t, 4)` |
| quantile regression | `nn.QuantileFFN(criterion=nn.metrics.QuantileLoss(alpha=0.1))` | output `(n, t, 2)`: mean, interval width |
| binary classification | `nn.BinaryClassificationFFN(n_tasks=t)` | do not normalize targets; outputs probabilities |
| multiclass | `nn.MulticlassClassificationFFN(n_classes=c, n_tasks=t)` | targets are class indices; output `(n, t, c)` |
| spectra | `nn.SpectralFFN(n_tasks=bins)` | |

Set `input_dim=mp.output_dim` on the predictor whenever the message-passing output size is not
300, and add the size of any extra molecule descriptors.

## Fine-tune CheMeleon from Python

```python
from pathlib import Path
from urllib.request import urlretrieve

import torch
from chemprop import featurizers, models, nn

path = Path.home() / ".chemprop" / "chemeleon_mp.pt"  # same cache location as the CLI
if not path.exists():
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(".part")
    urlretrieve("https://zenodo.org/records/15460715/files/chemeleon_mp.pt", partial)
    partial.rename(path)

chemeleon = torch.load(path, weights_only=True)
mp = nn.BondMessagePassing(**chemeleon["hyper_parameters"])
mp.load_state_dict(chemeleon["state_dict"])

# CheMeleon expects the default featurizer
featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()
agg = nn.MeanAggregation()
ffn = nn.RegressionFFN(input_dim=mp.output_dim)  # add output_transform=... from your target scaler
model = models.MPNN(mp, agg, ffn)
```

Then build datasets and train exactly as above. Do not hardcode the hidden size; use
`mp.output_dim`.

## Reactions

```python
import pandas as pd
from chemprop import data, featurizers, models, nn

df = pd.read_csv("reactions.csv")
all_data = [
    data.ReactionDatapoint.from_smi(rxn, y, keep_h=True)  # keep_h for explicit mapped hydrogens
    for rxn, y in zip(df["rxn"], df[["target"]].values)
]
featurizer = featurizers.CondensedGraphOfReactionFeaturizer(mode_="REAC_DIFF")  # note: mode_
dset = data.ReactionDataset(all_data, featurizer)

mp = nn.BondMessagePassing(*featurizer.shape)  # CGR feature sizes differ from the defaults
model = models.MPNN(mp, nn.MeanAggregation(), nn.RegressionFFN(input_dim=mp.output_dim))
```

Split reactions with `data.make_split_indices([d.rct for d in all_data], ...)`.

## Mixtures (multiple molecules per row)

```python
import pandas as pd
from chemprop import data, featurizers, models, nn

df = pd.read_csv("mixtures.csv")
columns = ["solute", "solvent"]
ys = df[["target"]].values
# Targets go on the first component only.
all_data = [[data.MoleculeDatapoint.from_smi(smi, y) for smi, y in zip(df[columns[0]], ys)]]
all_data += [[data.MoleculeDatapoint.from_smi(smi) for smi in df[col]] for col in columns[1:]]

featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()
dset = data.MulticomponentDataset([data.MoleculeDataset(d, featurizer) for d in all_data])

mcmp = nn.MulticomponentMessagePassing(
    blocks=[nn.BondMessagePassing() for _ in columns], n_components=len(columns)
)
model = models.MulticomponentMPNN(
    mcmp, nn.MeanAggregation(), nn.RegressionFFN(input_dim=mcmp.output_dim)
)
```

`data.split_data_by_indices` on a list of components returns `[replicate][component][datapoint]`.

## Extra features and descriptors

- Molecule descriptors: `MoleculeDatapoint(..., x_d=array)` or `from_smi(smi, y, x_d=array)`;
  scale with `X_d_scaler = train_dset.normalize_inputs("X_d")` and
  `val_dset.normalize_inputs("X_d", X_d_scaler)` (not the test set), pass
  `X_d_transform=nn.ScaleTransform.from_standard_scaler(X_d_scaler)` to `models.MPNN`, and
  increase the predictor's `input_dim` by the descriptor length. The transform is only applied
  at test and prediction time, which is why the validation set must be scaled in advance.
- Atom/bond features before message passing: `V_f`/`E_f` on the datapoint, with
  `SimpleMoleculeMolGraphFeaturizer(extra_atom_fdim=..., extra_bond_fdim=...)` and
  `nn.BondMessagePassing(*featurizer.shape)`.
- Atom descriptors after message passing: `V_d` on the datapoint and `d_vd=...` on the message
  passing.
- Built-in molecule featurizers (`featurizers.MorganBinaryFeaturizer()`,
  `featurizers.RDKit2DFeaturizer()`, ...) are called on an RDKit `Mol` to produce `x_d`.

See the example notebooks in the repository (`examples/`) for complete versions of each workflow:
https://github.com/chemprop/chemprop/tree/main/examples
