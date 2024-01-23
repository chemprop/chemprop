from argparse import ArgumentParser
import logging
import sys
from pathlib import Path

from chemprop.cli.train import TrainSubcommand
from chemprop.cli.predict import PredictSubcommand
from chemprop.cli.convert import ConvertSubcommand

# TODO: add subcommands for Fingerprint and Hyperopt
# from chemprop.cli.fingerprint import FingerprintSubcommand
# from chemprop.cli.hyperopt import HyperoptSubcommand

from chemprop.cli.utils import LOG_DIR, NOW, pop_attr

logger = logging.getLogger(__name__)

LOG_LEVELS = [logging.ERROR, logging.WARNING, logging.INFO, logging.DEBUG]
SUBCOMMANDS = [
    TrainSubcommand,
    PredictSubcommand,
    ConvertSubcommand,
]  # , FingerprintSubcommand, HyperoptSubcommand]


def main():
    parser = ArgumentParser()
    subparsers = parser.add_subparsers(title="mode", dest="mode", required=True)

    parent = ArgumentParser(add_help=False)
    parent.add_argument(
        "--logfile",
        "--log",
        nargs="?",
        const="default",
        help=f"the path to which the log file should be written. Specifying just the flag (i.e., '--log/--logfile') will automatically log to a file '{LOG_DIR}/MODE/{NOW}.log', where 'MODE' is the CLI mode chosen.",
    )
    parent.add_argument("-v", "--verbose", action="count", default=0, help="the verbosity level")

    parents = [parent]
    for subcommand in SUBCOMMANDS:
        subcommand.add(subparsers, parents)

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
