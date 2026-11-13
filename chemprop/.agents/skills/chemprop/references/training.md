# Training reference (`chemprop train`)

`chemprop train --help` lists every option with its default. This file covers what `--help` does
not make obvious.

## Task types

| `--task-type` | Targets | Default metric |
|---|---|---|
| `regression` | real values | `mse` |
| `regression-mve` / `regression-evidential` / `regression-quantile` | real values, with uncertainty (see `references/uncertainty.md`) | `mse` |
| `classification` / `classification-dirichlet` | 0/1 per task | `roc` |
| `multiclass` / `multiclass-dirichlet` | integer class index; set `--multiclass-num-classes` (default 3) | `multiclass-mcc` |
| `spectral` | one column per spectrum bin | `sid` (default only; not accepted by `--metrics`) |

```bash
chemprop train --data-path classification.csv --smiles-columns smiles --target-columns active \
    --task-type classification --metrics roc prc --epochs 50 --patience 10 \
    --output-dir runs/active
```

- Multitask: several `--target-columns`; empty cells are masked. `--show-individual-scores`
  prints per-task scores.
- Censored targets such as `<5.2` or `>7.0` need a bounded loss and metric:
  `--loss-function bounded-mse --metrics bounded-mse`.
- `--metrics` replaces the default metric list; include the default if you still want it.
- `--tracking-metric` (default `val_loss`) controls checkpointing and early stopping; it must be
  `val_loss` or one of `--metrics` (or the task's default metric when `--metrics` is not given).
  Hyphenated names are currently rejected as tracking metrics, including `binary-mcc` and
  `multiclass-mcc` (the multiclass default), so use `val_loss` for those.

## Splits

- `--split-type` (`--split-sizes` defaults to `0.8 0.1 0.1`):
  - `random` (default) and `random_with_repeated_smiles` (keeps duplicate SMILES together).
  - `kmeans`: k-means clusters (Euclidean) of 2048-bit Morgan fingerprints are kept together.
  - `kennard_stone`: maximally diverse training set in Morgan-fingerprint space.
  - `scaffold_balanced`: Bemis-Murcko scaffolds are kept together. Different scaffolds can still
    be very similar molecules, so this is not always a stringent test.
- No split is right for every use; choose based on how the model will be used, and use the
  user's preferred split if they have one.
- Predefined splits: `--splits-column COLUMN` (values `train`/`val`/`test`), or
  `--splits-file splits.json` (a list of `{"train": [...], "val": [...], "test": [...]}` index
  lists; each entry is one replicate).
- Separate files: `--data-path train.csv val.csv test.csv`, or `--data-path trainval.csv test.csv`
  with `--split-sizes 0.9 0.1 0.0` (the test fraction must be 0 when a test file is given).
- `--data-seed` controls the split; `--pytorch-seed` controls initialization (random by default).

## Replicates and ensembles

- `--num-replicates N` trains on N different splits (in `replicate_0/`, ...); with
  `kennard_stone`, which is deterministic, all replicates get the same split. Use it to report
  the mean and spread of test performance.
- `--ensemble-size M` trains M models on the same split (in `model_0/` ... `model_{M-1}/`).

```bash
chemprop train --data-path data.csv --smiles-columns smiles --target-columns target \
    --split-type kmeans --num-replicates 3 --ensemble-size 2 \
    --epochs 50 --patience 10 --output-dir runs/target_replicates
```

`chemprop predict --model-paths DIR` averages every `*.pt` file below `DIR`, including every
replicate. To use one replicate's ensemble, pass that replicate's directory.

## Architecture defaults that differ from v1

- `--ffn-num-layers` counts hidden layers only (v1 also counted the output layer).
- `--aggregation` defaults to `norm` (sum divided by `--aggregation-norm`, default 100).
- For mixtures, `--depth` and `--message-hidden-dim` can take one value per component, or use
  `--mpn-shared` to share one network.

## Transfer learning

```bash
chemprop train --data-path data.csv --smiles-columns smiles --target-columns target \
    --checkpoint runs/target_replicates/replicate_0/model_0/best.pt --freeze-encoder \
    --epochs 50 --patience 10 --output-dir runs/target_finetuned
```

- `--checkpoint` loads a whole trained model: the architecture, loss, metrics, learning rates,
  warmup, and scalers all come from the checkpoint, and the matching flags are ignored, but
  `--epochs` still sets how long the learning-rate schedule runs.
  `--dropout` is the exception: it is applied to the loaded model, and its default of 0 turns
  dropout off. One model is fine-tuned per checkpoint found.
- `--freeze-encoder` freezes message passing; `--frzn-ffn-layers N` also freezes the first N FFN
  layers.
- `--from-foundation CheMeleon` initializes only message passing; not combinable with
  `--checkpoint`.

## Hyperparameter optimization

Needs `pip install "chemprop[hpopt]"`. `chemprop hpopt` takes every `chemprop train` argument:

```text
chemprop hpopt --data-path data.csv --smiles-columns smiles --target-columns target \
    --epochs 50 --search-parameter-keywords basic --raytune-num-samples 20 \
    --hpopt-save-dir runs/hpopt
chemprop train --config-path runs/hpopt/best_config.toml --output-dir runs/target_best
```

- `--search-parameter-keywords`: `basic` (default: depth, hidden sizes, FFN layers, dropout),
  `learning_rate` or `all` (both need `--epochs 6` or more), or individual names such as `depth`.
- The search uses only the first split. The default search algorithm is `hyperopt`.
- If Ray fails with an "AF_UNIX path length" error, pass a short `--raytune-temp-dir`.

## Output layout

```text
OUTPUT_DIR/
  config.toml            # all arguments; rerun with --config-path
  splits.json            # split indices (not written when train/val/test files are given)
  replicate_0/           # only when --num-replicates > 1
    model_0/
      best.pt            # use for predict, fingerprint, --checkpoint
      test_predictions.csv
      checkpoints/       # Lightning .ckpt files; --remove-checkpoints deletes them
      trainer_logs/      # TensorBoard if installed, otherwise CSV
```
