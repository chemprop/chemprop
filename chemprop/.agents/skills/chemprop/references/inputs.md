# Input data reference

Every input-related flag used in `chemprop train` must be repeated, with the same settings, in
`chemprop predict` and `chemprop fingerprint`. Those commands do not read the training config.
Feature and descriptor files (`--descriptors-path`, `--atom-features-path`, ...) must be computed
for the new input rows, not reused from training.

## Mixtures and multiple molecules (e.g. solute + solvent)

Give one SMILES column per component. Each component gets its own message-passing network
(or one shared network with `--mpn-shared`), and the embeddings are concatenated.

```bash
chemprop train --data-path mixtures.csv --smiles-columns solute solvent \
    --target-columns target --epochs 50 --patience 10 --output-dir runs/mixtures
chemprop predict --test-path mixtures.csv --smiles-columns solute solvent \
    --model-paths runs/mixtures/model_0/best.pt --preds-path runs/mixtures/preds.csv
```

- The order of `--smiles-columns` must be the same for training and prediction.
- `--split-key-molecule 1` makes structure-based splits (`kmeans`, `scaffold_balanced`, ...) use
  the second component.
- Per-component architecture: `--depth 3 2`, `--message-hidden-dim 300 100`.

## Reactions

Reactions must be atom-mapped reaction SMILES, `reactants>agents>products` (agents optional), for
example `[CH3:1][OH:2]>>[CH2:1]=[O:2]`. Chemprop builds a condensed graph of reaction (CGR).

```bash
chemprop train --data-path reactions.csv --reaction-columns rxn --target-columns target \
    --rxn-mode reac_diff --keep-h --epochs 50 --patience 10 --output-dir runs/reactions
chemprop predict --test-path reactions.csv --reaction-columns rxn --rxn-mode reac_diff --keep-h \
    --model-paths runs/reactions/model_0/best.pt --preds-path runs/reactions/preds.csv
```

- `--rxn-mode`: `reac_diff` (default), `reac_prod`, `prod_diff`, each with a `_balance` variant
  for unbalanced reactions.
- Use `--keep-h` when the reaction SMILES contain explicit hydrogen atoms such as `[H:5]`;
  hydrogen counts inside brackets (`[CH3:1]`) do not need it.
- Reaction plus molecule (e.g. a solvent): `--reaction-columns rxn --smiles-columns solvent`.
  Components are always ordered molecules first, then reactions, whatever order the flags are
  given in, so per-component options (`--depth A B`, `--split-key-molecule`, the index in
  `--atom-features-path N file.npz`) count `--smiles-columns` before `--reaction-columns`.
- Structure-based splits on reactions use the reactants.

## Extra molecule-level features (descriptors)

These are concatenated to the learned embedding after aggregation.

- Computed by Chemprop: `--molecule-featurizers` with any of `morgan_binary`, `morgan_count`
  (radius 2, 2048 bits), `rdkit_2d`, `v1_rdkit_2d`, `v1_rdkit_2d_normalized`, `charge`. Morgan
  fingerprints should not be standardized; add `--no-descriptor-scaling`.
- From CSV columns, e.g. experimental conditions such as temperature:

```bash
chemprop train --data-path conditions.csv --smiles-columns smiles --target-columns target \
    --descriptors-columns temperature --epochs 50 --patience 10 --output-dir runs/conditions
chemprop predict --test-path conditions.csv --smiles-columns smiles \
    --descriptors-columns temperature \
    --model-paths runs/conditions/model_0/best.pt --preds-path runs/conditions/preds.csv
```

  Always pass `--target-columns` with `--descriptors-columns`, or the descriptor columns are also
  treated as targets.
- From a file: `--descriptors-path descriptors.npz`, saved with `np.savez(path, X_d)` where `X_d`
  has shape (number of rows, number of descriptors) in the same row order as the CSV.
- Descriptors are standardized by default; turn this off with `--no-descriptor-scaling`, for
  example when you scaled them yourself. RDKit 2D descriptors are not well served by standard
  scaling; precomputing and scaling them yourself is often better.

## Extra atom and bond features

Saved as `.npz` files with one array per molecule, in CSV row order, atoms and bonds in RDKit order:
`np.savez("atom_features.npz", *V_fs)` with each array of shape (atoms, features).

- `--atom-features-path`: concatenated to atom features before message passing.
- `--atom-descriptors-path`: concatenated to atom embeddings after message passing.
- `--bond-features-path`: concatenated to bond features before message passing.
- For mixtures, prefix the component index: `--atom-features-path 0 solute_atoms.npz`.
- Scaling switches: `--no-atom-feature-scaling`, `--no-atom-descriptor-scaling`,
  `--no-bond-feature-scaling`.
- These files only work with a single `--data-path` file. With `--from-foundation CheMeleon`,
  `--atom-features-path`, `--atom-descriptors-path`, and `--bond-features-path` are not allowed.

## Atom and bond featurization options

- `--multi-hot-atom-featurizer-mode`: `V2` (default), `V1` (needed for models converted from
  v1), `ORGANIC` (smaller, for organic molecules), `RIGR` (resonance-invariant).
- `--keep-h` keeps explicit hydrogens from the SMILES; `--add-h` adds hydrogens to every molecule.
- `--ignore-stereo` drops stereochemistry; `--reorder-atoms` reorders atoms by atom-map number.

## Atom- and bond-level targets

Predict per-atom and per-bond properties (optionally together with molecule-level targets). In
the CSV, atom and bond targets are Python-style lists, one value per atom or bond in RDKit order,
for example `"[0.1, -0.3, 0.2]"`.

```bash
chemprop train --data-path atoms_bonds.csv --smiles-columns smiles \
    --mol-target-columns mol_target --atom-target-columns atom_target \
    --bond-target-columns bond_target --keep-h --reorder-atoms \
    --epochs 50 --patience 10 --output-dir runs/atoms_bonds
```

- Only one SMILES column; no reactions or mixtures. `--from-foundation` is ignored for these models.
- The per-atom and per-bond values must follow the atom order of the parsed molecule:
  `--keep-h` keeps explicit hydrogens as atoms, and `--reorder-atoms` orders atoms by their
  atom-map numbers (only useful with atom-mapped SMILES). Use the same flags in `predict`.
- `--constraints-path` and `--constraints-to-targets` force atom or bond predictions to sum to a
  given value per molecule (for example, partial charges summing to the total charge).
- With several models, `predict` writes the averaged predictions plus one `<preds stem>_model_<i>`
  file per model (instead of `_individual.csv`).

## Sample weights and missing values

- `--weight-column` (`-w`): per-row weights in the loss.
- Empty cells in target columns are missing values and are ignored in the loss and metrics.
- `--ignore-columns` excludes columns from the default target list.
- Rows with SMILES that RDKit cannot parse cause an error; clean the data first.
