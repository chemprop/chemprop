# AGENTS.md

## Project

Chemprop v2.2.3 — message passing neural networks for molecular property prediction. CLI tool + Python library backed by PyTorch Lightning.

## Quick Start

```
pip install -e ".[dev,test]"
pytest tests                          # runs unit + CLI tests with coverage
pytest tests -m integration           # runs integration tests (overfit checks, slower)
```

## Commands

| Task | Command |
|------|---------|
| Install editable | `pip install -e ".[dev,test]"` |
| Format check | `black --check .` |
| Format fix | `black .` |
| Import sort check | `isort --check .` |
| Import sort fix | `isort .` |
| Lint | `flake8 .` |
| Run tests | `pytest` (runs with `--cov chemprop` by default) |
| Tests without coverage | `pytest --no-cov` |
| Only unit tests | `pytest tests/unit` |
| Only CLI tests | `pytest tests/cli` |
| Only integration tests | `pytest tests/integration` |
| Single test file | `pytest tests/unit/nn/test_metrics.py -v` |
| Single test by name | `pytest tests/unit/nn/test_metrics.py::test_name -v` |
| Build package | `python -m build .` |

CI pipeline order: build -> lint (black, flake8, isort) -> test.

## Formatting Rules

- **black**: v23 required, `line-length = 100`, `skip-magic-trailing-comma = true`, target py311
- **isort**: profile = "black", `line_length = 100`, `force_sort_within_sections = true`
- **flake8**: `max-line-length = 100`, ignores E203 E266 E501 F403 E741 W503 W605; `__init__.py` ignores F401; see `.flake8` for per-file ignores

## Architecture

```
chemprop/
  cli/          # CLI subcommands: train, predict, convert, fingerprint, hpopt
  data/         # Dataset/Datapoint classes, dataloaders, splits
  featurizers/  # Molecule/reaction featurization (RDKit-based)
  models/       # MPNN, MolAtomBondMPNN, MulticomponentMPNN
  nn/           # Message passing layers, FFNs, aggregations, metrics, predictors
  schedulers/   # Learning rate schedulers
  uncertainty/  # Uncertainty quantification methods
  utils/        # Shared utilities
```

Entrypoint: `chemprop.cli.main:main` (registered as `chemprop` console script).

## Testing Gotchas

- `tests/conftest.py` defines shared fixtures: `smis`, `mols`, `targets`, `data_dir`, regression/classification data fixtures
- `tests/integration/conftest.py` defines session-scoped MPNN model fixtures (parametrized by message passing type + aggregation)
- Integration tests verify overfitting on small datasets — they train real models, expect `accelerator="cpu"`
- Pre-trained model files (`*.pt`, `*.ckpt`) live in `tests/data/` and are gitignored except under `!tests/data/*` in `.gitignore`. Regenerate with `tests/regenerate_models.sh <conda_env> <repo_path>`.
- pytest markers: `integration`, `CLI`
- Default pytest `addopts = "--cov chemprop"` — use `--no-cov` when running integration tests or notebooks for speed

## Python Constraints

- `requires-python = ">=3.11,<3.15"` — supports 3.11, 3.12, 3.13, 3.14
- Core deps: `torch >= 2.1`, `lightning >= 2.0`, `rdkit`, `astartes[molecules]`

## Known Inconsistency

The edge update function in all versions uses preactivation instead of postactivation initial edge hidden states (documented in v2 paper SI).
