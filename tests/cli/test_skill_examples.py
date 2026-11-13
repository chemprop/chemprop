"""This tests that the commands and code in the bundled agent skill
(chemprop/.agents/skills/chemprop) run as written, so the skill does not go stale when the CLI or
API changes.

Every ``chemprop ...`` line in a ```bash block and every ```python block is run in order, in a
temporary directory containing small versions of the files the examples refer to, and the output
files the commands name are checked to exist. ```text blocks are not run. The CheMeleon examples
download the foundation model, so these tests need network access.
"""

from importlib.resources import files
from pathlib import Path
import re
import shlex

import pandas as pd
import pytest
import yaml

from chemprop.cli.main import main

pytestmark = pytest.mark.CLI

SKILL_DIR = Path(str(files("chemprop"))) / ".agents" / "skills" / "chemprop"
SKILL_FILES = sorted(p.relative_to(SKILL_DIR).as_posix() for p in SKILL_DIR.rglob("*.md"))
FENCED_BLOCK = re.compile(r"^ *```(\w*)\n(.*?)^ *```", re.MULTILINE | re.DOTALL)
LANGUAGES = {"bash", "python", "text"}
# keep the examples realistic in the skill, but fast and on CPU in the tests
TRAIN_OVERRIDES = ["--epochs", "3", "--warmup-epochs", "1"]
COMMON_OVERRIDES = ["--accelerator", "cpu"]
PYTHON_OVERRIDES = {r"max_epochs=\d+": "max_epochs=3", r'accelerator="auto"': 'accelerator="cpu"'}


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def get_blocks(path: Path) -> list[tuple[str, str]]:
    return FENCED_BLOCK.findall(read(path))


def get_commands(block: str) -> list[list[str]]:
    lines = block.replace("\\\n", " ").splitlines()
    return [shlex.split(line, comments=True) for line in lines if line.startswith("chemprop ")]


def get_option(args: list[str], *names: str) -> str | None:
    for name in names:
        if name in args:
            return args[args.index(name) + 1]
    return None


def check_outputs(args: list[str]):
    """Check that the output files a command names were written."""
    match args[1]:
        case "train":
            output_dir = Path(get_option(args, "--output-dir", "-o", "--save-dir"))
            assert (output_dir / "config.toml").is_file()
            assert list(output_dir.rglob("best.pt"))
        case "predict":
            assert Path(get_option(args, "--preds-path", "--output", "-o")).is_file()
        case "fingerprint":
            output = Path(get_option(args, "--output", "-o"))
            assert (output.parent / f"{output.stem}_0{output.suffix}").is_file()
        case "convert":
            assert Path(get_option(args, "--output-path", "-o")).is_file()


@pytest.fixture
def example_dir(data_dir, tmp_path, monkeypatch):
    def rename(df, columns):
        return df[list(columns)].rename(columns=columns)

    data = rename(
        pd.read_csv(data_dir / "regression/mol/mol.csv"), {"smiles": "smiles", "lipo": "target"}
    )
    data.to_csv(tmp_path / "data.csv", index=False)
    pd.read_csv(data_dir / "smis.csv").to_csv(tmp_path / "new.csv", index=False)
    data.assign(temperature=[290 + i % 20 for i in range(len(data))]).to_csv(
        tmp_path / "conditions.csv", index=False
    )
    rename(
        pd.read_csv(data_dir / "classification/mol.csv"), {"smiles": "smiles", "NR-AhR": "active"}
    ).to_csv(tmp_path / "classification.csv", index=False)
    rename(
        pd.read_csv(data_dir / "regression/mol+mol/mol+mol.csv"),
        {"smiles": "solute", "solvent": "solvent", "peakwavs_max": "target"},
    ).to_csv(tmp_path / "mixtures.csv", index=False)
    rename(
        pd.read_csv(data_dir / "regression/rxn/rxn.csv"), {"smiles": "rxn", "ea": "target"}
    ).to_csv(tmp_path / "reactions.csv", index=False)
    mab = rename(
        pd.read_csv(data_dir / "mol_atom_bond/regression.csv"),
        {
            "smiles": "smiles",
            "mol_y1": "mol_target",
            "atom_y1": "atom_target",
            "bond_y1": "bond_target",
        },
    )
    pd.concat([mab] * 10).to_csv(tmp_path / "atoms_bonds.csv", index=False)
    (tmp_path / "old_v1_model.pt").write_bytes(
        (data_dir / "example_model_v1_regression_mol.pt").read_bytes()
    )

    monkeypatch.chdir(tmp_path)
    return tmp_path


def test_skill_metadata():
    assert "SKILL.md" in SKILL_FILES, f"no skill found at {SKILL_DIR}"
    text = read(SKILL_DIR / "SKILL.md")
    frontmatter = re.match(r"---\n(.*?)\n---\n", text, re.DOTALL)
    assert frontmatter is not None

    # constraints from the Agent Skills specification, https://agentskills.io/specification
    fields = yaml.safe_load(frontmatter.group(1))
    assert fields["name"] == SKILL_DIR.name == "chemprop"
    assert re.fullmatch(r"[a-z0-9]+(-[a-z0-9]+)*", fields["name"])
    assert 0 < len(fields["description"]) <= 1024
    assert len(text.splitlines()) < 500


@pytest.mark.parametrize("skill_file", SKILL_FILES)
def test_skill_references_exist(skill_file):
    references = re.findall(r"`((?:references/)?[\w-]+\.md)`", read(SKILL_DIR / skill_file))
    for reference in references:
        assert reference.startswith("references/") or reference == "SKILL.md", reference
        assert (SKILL_DIR / reference).is_file(), reference


@pytest.mark.parametrize("skill_file", SKILL_FILES)
def test_skill_examples(skill_file, example_dir, monkeypatch):
    for language, block in get_blocks(SKILL_DIR / skill_file):
        # every code block must be tagged so it is either run here or explicitly skipped
        assert language in LANGUAGES, f"untagged or unknown code block in {skill_file}:\n{block}"
        if language == "bash":
            commands = get_commands(block)
            assert commands, f"bash block without chemprop commands in {skill_file}:\n{block}"
            for command in commands:
                args = command + (TRAIN_OVERRIDES if command[1] == "train" else [])
                args += (
                    COMMON_OVERRIDES if command[1] in ("train", "predict", "fingerprint") else []
                )
                with monkeypatch.context() as m:
                    m.setattr("sys.argv", args)
                    try:
                        main()
                        check_outputs(args)
                    except BaseException as e:
                        e.add_note(f"Failed command from {skill_file}: {shlex.join(command)}")
                        raise

        elif language == "python":
            code = block
            for pattern, replacement in PYTHON_OVERRIDES.items():
                code = re.sub(pattern, replacement, code)
            try:
                exec(compile(code, f"{skill_file} (python block)", "exec"), {})
            except BaseException as e:
                e.add_note(f"Failed python block from {skill_file}:\n{block}")
                raise
