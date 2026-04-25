import logging
import os
from pathlib import Path
import sys

from configargparse import ArgumentParser

from chemprop.cli.conf import LOG_DIR, LOG_LEVELS, NOW
from chemprop.cli.convert import ConvertSubcommand
from chemprop.cli.fingerprint import FingerprintSubcommand
from chemprop.cli.hpopt import HpoptSubcommand
from chemprop.cli.predict import PredictSubcommand
from chemprop.cli.train import TrainSubcommand
from chemprop.cli.utils import pop_attr

logger = logging.getLogger(__name__)

SUBCOMMANDS = [
    TrainSubcommand,
    PredictSubcommand,
    ConvertSubcommand,
    FingerprintSubcommand,
    HpoptSubcommand,
]


def construct_parser():
    parser = ArgumentParser()
    subparsers = parser.add_subparsers(title="mode", dest="mode", required=True)

    parent = ArgumentParser(add_help=False)
    parent.add_argument(
        "--logfile",
        "--log",
        nargs="?",
        const="default",
        help=f"Path to which the log file should be written (specifying just the flag alone will automatically log to a file ``{LOG_DIR}/MODE/TIMESTAMP.log`` , where 'MODE' is the CLI mode chosen, e.g., ``{LOG_DIR}/MODE/{NOW}.log``)",
    )
    parent.add_argument("-v", action="store_true", help="Increase verbosity level to DEBUG")
    parent.add_argument(
        "-q",
        action="count",
        default=0,
        help="Decrease verbosity level to WARNING or ERROR if specified twice",
    )

    parents = [parent]
    for subcommand in SUBCOMMANDS:
        subcommand.add(subparsers, parents)

    return parser


def _resolve_log_path(logfile: str, mode: str) -> Path:
    if logfile == "default":
        (LOG_DIR / mode).mkdir(parents=True, exist_ok=True)
        return LOG_DIR / mode / f"{NOW}.log"
    log_path = Path(logfile)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    return log_path


def _safe_fileno(stream) -> int | None:
    try:
        return stream.fileno()
    except (AttributeError, OSError, ValueError):
        return None


def _redirect_streams_to(log_path: Path) -> dict:
    """Redirect Python and OS-level stdout/stderr to ``log_path``.

    Returns the state needed to restore them. The OS-level ``os.dup2`` is
    required so that Lightning's rich progress bar (which captures
    ``sys.stderr`` at import time) writes into the log file too. See
    chemprop#1357.

    If a stream has no underlying OS file descriptor (e.g. when pytest is
    capturing at ``sys`` level, or sys.stdout has been replaced with a
    StringIO), the fd-level redirect for that stream is skipped. Setup is
    rollback-safe: if any step fails, partial redirects are undone and the
    log file is closed before the exception propagates.
    """
    log_stream = open(log_path, "a", encoding="utf-8", errors="backslashreplace", buffering=1)

    saved_stdout_fd: int | None = None
    saved_stderr_fd: int | None = None
    saved_stdout, saved_stderr = sys.stdout, sys.stderr
    stdout_fd: int | None = None
    stderr_fd: int | None = None
    stdout_redirected = False
    stderr_redirected = False
    try:
        sys.stdout.flush()
        sys.stderr.flush()
        stdout_fd = _safe_fileno(sys.stdout)
        stderr_fd = _safe_fileno(sys.stderr)
        if stdout_fd is not None:
            saved_stdout_fd = os.dup(stdout_fd)
            os.dup2(log_stream.fileno(), stdout_fd)
            stdout_redirected = True
        if stderr_fd is not None:
            saved_stderr_fd = os.dup(stderr_fd)
            os.dup2(log_stream.fileno(), stderr_fd)
            stderr_redirected = True
        sys.stdout = log_stream
        sys.stderr = log_stream
    except BaseException:
        if stdout_redirected and saved_stdout_fd is not None:
            os.dup2(saved_stdout_fd, stdout_fd)
        if stderr_redirected and saved_stderr_fd is not None:
            os.dup2(saved_stderr_fd, stderr_fd)
        if saved_stdout_fd is not None:
            os.close(saved_stdout_fd)
        if saved_stderr_fd is not None:
            os.close(saved_stderr_fd)
        log_stream.close()
        raise

    return {
        "stream": log_stream,
        "saved_stdout": saved_stdout,
        "saved_stderr": saved_stderr,
        "saved_stdout_fd": saved_stdout_fd,
        "saved_stderr_fd": saved_stderr_fd,
        "stdout_fd": stdout_fd,
        "stderr_fd": stderr_fd,
    }


def _restore_streams(state: dict) -> None:
    log_stream = state["stream"]
    log_stream.flush()
    if state["saved_stdout_fd"] is not None:
        os.dup2(state["saved_stdout_fd"], state["stdout_fd"])
        os.close(state["saved_stdout_fd"])
    if state["saved_stderr_fd"] is not None:
        os.dup2(state["saved_stderr_fd"], state["stderr_fd"])
        os.close(state["saved_stderr_fd"])
    sys.stdout = state["saved_stdout"]
    sys.stderr = state["saved_stderr"]
    log_stream.close()


def main():
    parser = construct_parser()
    args = parser.parse_args()
    logfile, v_flag, q_count, mode, func = (
        pop_attr(args, attr) for attr in ["logfile", "v", "q", "mode", "func"]
    )

    if v_flag and q_count:
        parser.error("The -v and -q options cannot be used together.")

    # When --logfile is set, redirect output to the log file at both the
    # Python and OS level. The OS-level dup2 is required because Lightning's
    # ``rich``-based progress bar captures ``sys.stderr`` when its console
    # is constructed (at import time), so a pure ``sys.stderr = ...``
    # reassignment misses progress-bar output. See chemprop#1357.
    redirect_state = None
    handler = None
    if logfile is not None:
        log_path = _resolve_log_path(logfile, mode)
        redirect_state = _redirect_streams_to(log_path)

    try:
        if redirect_state is None:
            handler = logging.StreamHandler(sys.stderr)
        else:
            handler = logging.StreamHandler(redirect_state["stream"])

        verbosity = q_count * -1 if q_count else (1 if v_flag else 0)
        logging_level = LOG_LEVELS.get(verbosity, logging.ERROR)
        logging.basicConfig(
            handlers=[handler],
            format="%(asctime)s - %(levelname)s:%(name)s - %(message)s",
            level=logging_level,
            datefmt="%Y-%m-%dT%H:%M:%S",
            force=True,
        )

        logger.info(f"Running in mode '{mode}' with args: {vars(args)}")

        func(args)
    finally:
        if redirect_state is not None:
            # Detach and close the file-backed handler, restore streams, and
            # close the file even if anything above raised, so later
            # in-process invocations of main() start fresh.
            if handler is not None:
                logging.root.removeHandler(handler)
                handler.close()
            _restore_streams(redirect_state)
