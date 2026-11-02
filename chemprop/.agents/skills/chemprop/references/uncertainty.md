# Uncertainty reference (`chemprop predict`)

Uncertainty is estimated at prediction time, but most methods need a model trained with a
matching `--task-type`.

| `--uncertainty-method` | Model requirement |
|---|---|
| `mve` | trained with `--task-type regression-mve` |
| `evidential-total`, `evidential-epistemic`, `evidential-aleatoric` | `--task-type regression-evidential` |
| `quantile-regression` | `--task-type regression-quantile` |
| `classification` | `--task-type classification` (uses the predicted probability) |
| `classification-dirichlet` | `--task-type classification-dirichlet` |
| `multiclass-dirichlet` | `--task-type multiclass-dirichlet` |
| `ensemble` | two or more models passed to `--model-paths`, trained with plain `--task-type regression` or `classification` |
| `dropout` | a plain `--task-type regression` or `classification` model; Monte Carlo dropout (`--uncertainty-dropout-p` 0.1, `--dropout-sampling-size` 10) |

`ensemble` and `dropout` currently crash for multiclass models and for models trained with an
uncertainty task type (`regression-mve`, `regression-evidential`, `regression-quantile`,
`*-dirichlet`); use those models' own method instead.

Example: train a mean-variance model, then predict with calibrated uncertainty and evaluate it.

```bash
chemprop train --data-path data.csv --smiles-columns smiles --target-columns target \
    --task-type regression-mve --epochs 50 --patience 10 --output-dir runs/target_mve
chemprop predict --test-path data.csv --smiles-columns smiles \
    --model-paths runs/target_mve/model_0/best.pt --preds-path runs/target_mve/preds.csv \
    --uncertainty-method mve --cal-path data.csv --calibration-method zscaling \
    --evaluation-methods nll-regression miscalibration_area spearman
```

In practice `--cal-path` should be a held-out calibration set with true target values (for
example the validation split), not the training data, and the test set must contain the target
columns for `--evaluation-methods` to run. As in any prediction, the `target` column of
`preds.csv` then holds the predictions, so compare against the original file.

- Output: the predictions file gets a `<target>_unc` column per target, except with
  `--uncertainty-method classification`, which writes no `_unc` column (calibrated values then
  only appear in the logged evaluation). Multiclass models add `<target>_prob` columns. The
  meaning of `_unc` depends on the method and calibrator: a variance for uncalibrated `mve` or
  `ensemble`, the interval half-width for `quantile-regression` and `conformal-regression`, and a
  scaled variance for `zelikman-interval`.
- `--calibration-method` needs `--cal-path`:
  - regression: `zscaling`, `zelikman-interval` (uses `--calibration-interval-percentile`,
    default 95), `mve-weighting` (needs an ensemble of `regression-mve` models),
    `conformal-regression`.
  - binary classification: `platt`, `isotonic`, `conformal-multilabel`.
  - multiclass: `conformal-multiclass`, `conformal-adaptive`, `isotonic-multiclass`.
  - All conformal methods use `--conformal-alpha` (default 0.1).
- `--evaluation-methods`:
  - regression: `nll-regression`, `miscalibration_area`, `ence`, `spearman`,
    `conformal-coverage-regression`.
  - binary classification: `nll-classification`, `conformal-coverage-classification`.
  - multiclass: `nll-multiclass`, `conformal-coverage-multiclass`.
- Calibration with models that need external feature files does not currently work:
  `--cal-descriptors-path` and `--cal-atom-features-path` crash. Models using
  `--molecule-featurizers` calibrate fine, since the features are computed from the SMILES.

Ensemble uncertainty from an ensemble trained with `--ensemble-size`:

```bash
chemprop train --data-path data.csv --smiles-columns smiles --target-columns target \
    --ensemble-size 3 --epochs 50 --patience 10 --output-dir runs/target_ensemble
chemprop predict --test-path new.csv --smiles-columns smiles \
    --model-paths runs/target_ensemble --preds-path runs/target_ensemble/preds.csv \
    --uncertainty-method ensemble
```
