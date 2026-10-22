# Migrating from Chemprop v1

v2 is a rewrite. v1 scripts, flags, and Python imports do not work unchanged.

## Commands and flags

| v1 | v2 |
|---|---|
| `chemprop_train`, `chemprop_predict`, `chemprop_fingerprint`, `chemprop_hyperopt` | `chemprop train`, `chemprop predict`, `chemprop fingerprint`, `chemprop hpopt` |
| `chemprop_interpret` | `chemprop predict ... --callback myerson` (a different attribution method: Myerson values per atom) |
| `chemprop_web`, `sklearn_train`, `sklearn_predict` | no v2 equivalent |
| `--data_path` (underscores) | `--data-path` (hyphens everywhere) |
| `--dataset_type` | `--task-type` (`regression`, `classification`, `multiclass`, `spectral`, ...) |
| `--save_dir` | `--output-dir` |
| `--checkpoint_dir`, `--checkpoint_path` (predict) | `--model-paths` |
| `--checkpoint_dir`, `--checkpoint_path` (train, transfer learning) | `--checkpoint` |
| `--features_path` | `--descriptors-path` (`.npz`) |
| `--features_generator` | `--molecule-featurizers` (v1 RDKit features: `v1_rdkit_2d`, `v1_rdkit_2d_normalized`) |
| `--no_features_scaling` | `--no-descriptor-scaling` |
| `--num_folds` | `--num-replicates` |
| `--hidden_size` | `--message-hidden-dim` |
| `--ffn_num_layers 2` | `--ffn-num-layers 1` (v2 counts hidden layers only) |
| `--gpu 0` | `CUDA_VISIBLE_DEVICES=0` with `--accelerator gpu --devices 1` |
| `--metric auc` / `prc-auc` / `cross_entropy` / `mcc` | `--metrics roc` / `prc` / (`ce` is a loss, not a metric) / `binary-mcc` (or `multiclass-mcc` for multiclass) |
| `--number_of_molecules 2` | pass two names to `--smiles-columns` |
| `--reaction` + `--reaction_mode` | `--reaction-columns` + `--rxn-mode` |

## Converting v1 model files

```bash
chemprop convert --input-path old_v1_model.pt --output-path model_v2.pt
chemprop predict --test-path new.csv --smiles-columns smiles --model-paths model_v2.pt \
    --multi-hot-atom-featurizer-mode V1 --preds-path new_preds_from_v1_model.csv
```

- Converted v1 models use v1 atom features. `--multi-hot-atom-featurizer-mode V1` is optional
  for `predict` (it detects this and switches, with a warning) and required for `fingerprint`.
- If the v1 model was trained with `--features_generator`, pass the matching
  `--molecule-featurizers` (e.g. `v1_rdkit_2d_normalized`) when predicting.
- Models saved by Chemprop 2.0 can be upgraded with `chemprop convert --conversion v2_0_to_v2_1`.
- From Python: `chemprop.utils.v1_to_v2.convert_model_file_v1_to_v2(input_path, output_path)`.

## Python code

There is no v2 equivalent of `chemprop.args.TrainArgs`, `chemprop.train.run_training`,
`chemprop.train.make_predictions`, `chemprop.data.get_data`, `MoleculeModel`, or
`scaffold_split`. Rewrite using `references/python-api.md`:

- data loading: `data.MoleculeDatapoint.from_smi` + `data.MoleculeDataset` + `data.build_dataloader`
- splitting: `data.make_split_indices(mols, "scaffold_balanced", sizes)` +
  `data.split_data_by_indices`
- model: `models.MPNN(nn.BondMessagePassing(), nn.MeanAggregation(), nn.RegressionFFN())`
- training and prediction: a Lightning `Trainer` (`fit`, `test`, `predict`)

Renamed in v2.1 (code written for v2.0): the module `chemprop.nn.loss` became
`chemprop.nn.metrics`, and classes were renamed (`MSELoss`/`MSEMetric` → `MSE`, `SIDLoss` → `SID`,
`BinaryAUROCMetric` → `BinaryAUROC`, ...). `make_split_indices` no longer accepts `num_folds` and
returns one list per replicate.
