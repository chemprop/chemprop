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
        help=f"The path to which the log file should be written. Specifying just the flag (i.e., '--log/--logfile') will automatically log to a file '{LOG_DIR}/MODE/TIMESTAMP.log', where 'MODE' is the CLI mode chosen. An example 'TIMESTAMP' is {NOW}.",
    )
    parent.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="The verbosity level, specify the flag multiple times to increase verbosity.",
    )

    parents = [parent]
    for subcommand in SUBCOMMANDS:
        subcommand.add(subparsers, parents)

    return parser


def main():
    parser = construct_parser()
    args = parser.parse_args()
    logfile, verbose, mode, func = (
        pop_attr(args, attr) for attr in ["logfile", "verbose", "mode", "func"]
    )

    match logfile:
        case None:
            handler = logging.StreamHandler(sys.stderr)
        case "default":
            (LOG_DIR / mode).mkdir(parents=True, exist_ok=True)
            handler = logging.FileHandler(str(LOG_DIR / mode / f"{NOW}.log"))
        case _:
            Path(logfile).parent.mkdir(parents=True, exist_ok=True)
            handler = logging.FileHandler(logfile)

    logging.basicConfig(
        handlers=[handler],
        format="%(asctime)s - %(levelname)s:%(name)s - %(message)s",
        level=LOG_LEVELS[min(verbose, len(LOG_LEVELS) - 1)],
        datefmt="%Y-%m-%dT%H:%M:%S",
        force=True,
    )

    logger.info(f"Running in mode '{mode}' with args: {vars(args)}")

    func(args)
