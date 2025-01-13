import logging
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


def main():
    parser = construct_parser()
    args = parser.parse_args()
    logfile, v_flag, q_count, mode, func = (
        pop_attr(args, attr) for attr in ["logfile", "v", "q", "mode", "func"]
    )

    if v_flag and q_count:
        parser.error("The -v and -q options cannot be used together.")

    match logfile:
        case None:
            handler = logging.StreamHandler(sys.stderr)
        case "default":
            (LOG_DIR / mode).mkdir(parents=True, exist_ok=True)
            handler = logging.FileHandler(str(LOG_DIR / mode / f"{NOW}.log"))
        case _:
            Path(logfile).parent.mkdir(parents=True, exist_ok=True)
            handler = logging.FileHandler(logfile)

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
