---
name: chemprop
description: Train and use Chemprop v2 message-passing neural networks (D-MPNN graph neural networks) to predict molecular and reaction properties from SMILES (e.g. QSAR/ADMET properties such as solubility, lipophilicity, or toxicity) - regression, classification, multiclass, multitask, spectra, multicomponent mixtures (e.g. solute + solvent), atom-mapped reactions, atom/bond-level targets, uncertainty estimation, learned fingerprints, atom-level interpretation, hyperparameter optimization, and CheMeleon foundation-model fine-tuning. Use when the user wants a property-prediction model from SMILES or CSV data, wants to run, write, or debug a chemprop command or script, or mentions chemprop.
license: MIT
---

# Chemprop v2

Chemprop trains directed message-passing neural networks (D-MPNNs) on molecular graphs built
from SMILES. It has a CLI (`chemprop train|predict|fingerprint|hpopt|convert`) and a Python API
built on PyTorch Lightning. Prefer the CLI unless the user needs custom architectures, custom
training loops, or integration into existing Python code.

## Read this first: this is v2, not v1

Most examples online and in model training data are for Chemprop v1, which has a different CLI
and Python API. Do not write v1 code.

| v1 (wrong) | v2 (right) |
|---|---|
| `chemprop_train`, `chemprop_predict` | `chemprop train`, `chemprop predict` |
| `--dataset_type regression` / `--dataset-type` | `--task-type regression` (`-t`) |
| `--save_dir`, `--checkpoint_dir` | `--output-dir` (`-o`) for train, `--model-paths` for predict |
| `--features_path` | `--descriptors-path` (`.npz`) |
| `--features_generator rdkit_2d_normalized` | `--molecule-featurizers v1_rdkit_2d_normalized` |
| `--num_folds 5` | `--num-replicates 5` (`--num-folds` is an error) |
| `--gpu 0` | `CUDA_VISIBLE_DEVICES=0` with `--accelerator gpu --devices 1` (see `references/workflows.md`) |
| `from chemprop.args import TrainArgs`, `chemprop.train.run_training` | `from chemprop import data, featurizers, models, nn` + Lightning `Trainer` |

Only the subcommands `train`, `predict`, `fingerprint`, `hpopt`, and `convert` exist, and
`chemprop <subcommand> --help` lists every option. Old v1 models and scripts:
`references/v1-to-v2.md`.

## Install

`pip install chemprop` (Python 3.11 to 3.14; `pip install "chemprop[hpopt]"` adds `chemprop hpopt`).
For a specific CUDA or CPU-only PyTorch, install `torch` first. In a conda env with RDKit from
conda, install the dependencies with conda and then `pip install --no-deps chemprop`, or pip may
replace RDKit and cause crashes. Check with `chemprop --version`.

## Input data

A CSV with a header row, a SMILES column, and target columns; name them with `--smiles-columns` and
`--target-columns` (otherwise the first column is SMILES and all others are targets). Empty cells
are missing targets. Mixtures, reactions, descriptors, atom/bond targets: `references/inputs.md`.

## Train

```bash
chemprop train --data-path data.csv --smiles-columns smiles --target-columns target \
    --task-type regression --epochs 50 --patience 10 --output-dir runs/target
```

This splits the data 80/10/10 into train/validation/test, keeps the checkpoint with the lowest
validation loss, and evaluates it on the test split. The trained model is
`runs/target/model_0/best.pt`, and `config.toml` records every argument (rerun with
`--config-path`).

- Test metrics are only logged (lines like `test/mse: 0.71` on stderr), not saved to a file; add
  `--logfile runs/target/train.log` to keep them, and don't pass `-q`, which hides them.
- `model_0/test_predictions.csv` holds the test-split **predictions**, in columns named after the
  targets. The true values are in the input CSV, at the test indices listed in `splits.json`.

Other `--task-type` values: `classification`, `multiclass`, `spectral`, and uncertainty-aware
variants (`regression-mve`, `regression-evidential`, `regression-quantile`,
`classification-dirichlet`, `multiclass-dirichlet`). See `references/training.md`.

## Predict

```bash
chemprop predict --test-path new.csv --smiles-columns smiles \
    --model-paths runs/target/model_0/best.pt --preds-path runs/target/new_preds.csv
```

- The output is the input CSV plus one column per training target, named after the target
  (`target` here). **If the input already has a column with that name, it is overwritten with the
  predictions**, so keep true values under a different column name or in a separate file.
- `predict` does not read the training config. Repeat every input-related flag from training:
  `--smiles-columns`/`--reaction-columns`, `--descriptors-columns`, `--molecule-featurizers`,
  descriptor and feature files, `--rxn-mode`, `--multi-hot-atom-featurizer-mode`,
  `--keep-h`/`--add-h`.
- `--model-paths` accepts several files or a directory. A directory is searched recursively for
  `*.pt`, and all models found are averaged, with per-model predictions written to
  `<preds stem>_individual.csv`.
- Uncertainty estimates and calibration: `references/uncertainty.md`.
- Atom-level interpretation (which atoms drive a prediction): add `--callback myerson` to
  `chemprop predict` (options via `--callback-params '{"sampling_threshold": 20}'`). It works for
  `regression` and `classification` models, writes `<preds stem>_myerson_explanation.npz` (with
  `_0`, `_1`, ... per model if several are given), and is slow for molecules with many heavy atoms.

## Pitfalls

- **Always pass `--output-dir`.** The default is a new timestamped directory under
  `chemprop_training/` in the current directory on every run.
- **`--epochs` must be larger than `--warmup-epochs` (default 2).** `--epochs 1` alone fails.
- **Early stopping is off unless you pass `--patience`.** The default runs all `--epochs`
  (default 50) and keeps the best validation checkpoint.
- **The default split is random**, which usually gives an optimistic estimate for molecules
  unlike the training data. Pick the split to match how the model will be used and tell the user
  which one you chose: `--split-type kmeans` (k-means clusters of Morgan fingerprints),
  `kennard_stone`, or `scaffold_balanced` are stricter options, each with known limitations; if the
  user has a preferred split, use it (`--splits-column` or `--splits-file` for custom splits). Use
  `--num-replicates N` to see the variance across splits (not with `kennard_stone`, which gives the
  same split every time).
- **Pass `--target-columns` whenever you use `--descriptors-columns`**, otherwise the descriptor
  columns are also trained as targets.
- External feature files (`--descriptors-path`, `--atom-features-path`, ...) are only supported
  with a single `--data-path` file, not with separate train/val/test files.
- `--num-workers` above 0 can hang on Windows and macOS; the default is 0.
- The default metric is `mse` for regression, `roc` for classification, and `multiclass-mcc` for
  multiclass. `--metrics` replaces that default rather than adding to it, so include it if you
  still want it, e.g. `--metrics mse rmse mae r2`.
- Training is not reproducible run-to-run unless you pass `--pytorch-seed`.

## References

Read these only when the task needs them:

- `references/workflows.md`: CheMeleon fine-tuning (small datasets), learned fingerprints,
  GPUs and multi-GPU (DDP) training.
- `references/training.md`: task types, splits, replicates and ensembles, transfer learning,
  hyperparameter optimization (`chemprop hpopt`), output layout.
- `references/inputs.md`: mixtures, reactions, extra molecule/atom/bond features and descriptors,
  atom/bond-level targets, sample weights.
- `references/uncertainty.md`: uncertainty estimation, calibration, and evaluation with
  `chemprop predict`.
- `references/python-api.md`: training and predicting from Python, including CheMeleon.
- `references/v1-to-v2.md`: migrating v1 commands, scripts, and model files.

Full documentation: https://chemprop.readthedocs.io. When this skill and the installed version
disagree, `chemprop <subcommand> --help` is authoritative.
