# Workflows reference

CheMeleon fine-tuning, learned fingerprints, and GPUs.

## CheMeleon foundation model

CheMeleon is a pretrained message-passing model. On small datasets (roughly under a few thousand
molecules), fine-tuning it is often more accurate than training from random weights. The first
run downloads the pretrained weights from Zenodo to `~/.chemprop/`, so it needs network access.

```bash
chemprop train --data-path data.csv --smiles-columns smiles --target-columns target \
    --task-type regression --from-foundation CheMeleon \
    --epochs 50 --patience 10 --output-dir runs/target_chemeleon
```

It is not always the most accurate option: for some properties, molecular descriptors help more.
When accuracy matters, compare candidates on the same split (same `--split-type` and
`--data-seed`, or one `--splits-file`) and tell the user what you tried, for example:

```text
chemprop train ... --from-foundation CheMeleon
chemprop train ... --molecule-featurizers v1_rdkit_2d_normalized
chemprop train ... --from-foundation CheMeleon --molecule-featurizers v1_rdkit_2d_normalized
```

- CheMeleon is for molecule SMILES inputs (`--smiles-columns`), not `--reaction-columns`.
- Message-passing settings (`--depth`, `--message-hidden-dim`, `--aggregation`, ...) come from the
  foundation model. A warning listing them as ignored is printed on every run, even if you did not
  pass them; it is safe to ignore in that case.
- It cannot be combined with `--checkpoint`, a non-default `--multi-hot-atom-featurizer-mode`, or
  `--atom-features-path`/`--atom-descriptors-path`/`--bond-features-path`.

## Learned fingerprints

```bash
chemprop fingerprint --test-path new.csv --smiles-columns smiles \
    --model-paths runs/target_chemeleon/model_0/best.pt --ffn-block-index 0 \
    --output runs/target_chemeleon/new_fps.csv
```

`--ffn-block-index` is required: `0` is the learned molecule embedding right after aggregation
(with any extra molecule descriptors appended), `1` is the output of the first FFN layer, and so
on. One file is written per model, with `_0`, `_1`, ... appended to the name (here
`runs/target_chemeleon/new_fps_0.csv`), with columns `fp_0, fp_1, ...` and no SMILES column (rows follow the
input order).

## GPUs

`--accelerator auto --devices auto` (the defaults) use every GPU PyTorch can see, and with more
than one GPU that means multi-GPU training (DDP) whether or not you asked for it. On shared
machines, choose the GPUs explicitly:

```text
CUDA_VISIBLE_DEVICES=1 chemprop train ... --accelerator gpu --devices 1    # one specific GPU
CUDA_VISIBLE_DEVICES=0,1 chemprop train ... --accelerator gpu --devices 2  # DDP on two GPUs
chemprop train ... --accelerator cpu                                        # no GPU
```

- Check that training really runs on the GPU: the log prints `GPU available: True (cuda), used:
  True`. If it says `False`, check `nvidia-smi` and
  `python -c "import torch; print(torch.cuda.is_available())"` (a CPU-only PyTorch build is the
  usual cause).
- DDP runs one process per GPU, so log lines (including `test/...` metrics) appear once per GPU,
  and each process uses `--batch-size`, so the effective batch size is that times the number of
  GPUs. Outputs (`best.pt`, `test_predictions.csv`) are written once, as for a single GPU.
- When calling Chemprop in-process (a notebook, a test, or `chemprop.cli.main.main()`), use a
  single device (`--devices 1` or `pl.Trainer(devices=1)`); multi-process DDP can fail there.

