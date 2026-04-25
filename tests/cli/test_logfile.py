"""Tests for ``--logfile`` stream handling (chemprop#1357)."""

import logging
import sys

import pytest

from chemprop.cli.main import main


@pytest.fixture
def data_path(data_dir):
    return str(data_dir / "regression" / "mol" / "mol.csv")


def _train_args(input_path, save_dir, extra=()):
    return [
        "chemprop",
        "train",
        "-i",
        input_path,
        "--epochs",
        "3",
        "--num-workers",
        "0",
        "--save-dir",
        str(save_dir),
        *extra,
    ]


@pytest.mark.CLI
def test_logfile_captures_stderr_writes_through_main(
    monkeypatch, tmp_path, data_path, capfd
):
    """A run with --logfile must write all stderr output (including
    Lightning's rich-based banner that historically leaked to the console)
    to the file and leave the console silent. This is the regression
    reported in chemprop#1357."""
    log_path = tmp_path / "run.log"
    args = _train_args(data_path, tmp_path / "out", ("--logfile", str(log_path)))

    capfd.readouterr()  # discard anything captured before this test
    with monkeypatch.context() as m:
        m.setattr("sys.argv", args)
        main()
    captured = capfd.readouterr()

    assert log_path.exists() and log_path.stat().st_size > 0
    content = log_path.read_text()
    assert "Running in mode 'train'" in content
    # Lightning's rank-zero banner is written to ``sys.stderr`` via a
    # rich.Console captured at import time; without the fd-level redirect
    # it would still appear on the terminal even when --logfile is set.
    assert "GPU available" in content or "TPU available" in content
    # And the terminal must stay silent.
    assert captured.out == ""
    assert captured.err == ""


@pytest.mark.CLI
def test_logfile_restores_streams_after_main(monkeypatch, tmp_path, data_path):
    """main() must restore sys.stdout/sys.stderr so a later invocation without
    --logfile does not keep writing to the previous log file."""
    log_path = tmp_path / "run.log"
    saved_stdout, saved_stderr = sys.stdout, sys.stderr
    args = _train_args(data_path, tmp_path / "out", ("--logfile", str(log_path)))

    with monkeypatch.context() as m:
        m.setattr("sys.argv", args)
        main()

    assert sys.stdout is saved_stdout
    assert sys.stderr is saved_stderr


@pytest.mark.CLI
def test_logfile_then_no_logfile_does_not_pollute(monkeypatch, tmp_path, data_path):
    """Reentrancy: a second main() call without --logfile must not append to
    the first run's log file."""
    log_path = tmp_path / "first.log"
    args1 = _train_args(data_path, tmp_path / "out1", ("--logfile", str(log_path)))
    args2 = _train_args(data_path, tmp_path / "out2")

    with monkeypatch.context() as m:
        m.setattr("sys.argv", args1)
        main()

    size_after_first = log_path.stat().st_size

    with monkeypatch.context() as m:
        m.setattr("sys.argv", args2)
        main()

    assert log_path.stat().st_size == size_after_first


@pytest.mark.CLI
def test_logfile_default_branch_creates_file(monkeypatch, tmp_path, data_path):
    """The bare ``--log`` form (logfile=='default') must produce a writable
    log file under LOG_DIR/<mode>/."""
    from chemprop.cli.conf import LOG_DIR

    args = _train_args(data_path, tmp_path / "out", ("--log",))

    before = {p for p in (LOG_DIR / "train").glob("*.log")} if (LOG_DIR / "train").exists() else set()
    with monkeypatch.context() as m:
        m.setattr("sys.argv", args)
        main()
    after = {p for p in (LOG_DIR / "train").glob("*.log")}
    new_logs = after - before

    assert new_logs, "expected --log to create a new log file under LOG_DIR/train/"
    new_log = next(iter(new_logs))
    assert "Running in mode 'train'" in new_log.read_text()
    new_log.unlink()
