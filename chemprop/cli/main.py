from argparse import ArgumentParser
import logging
from pathlib import Path

from chemprop.cli.train import TrainSubcommand
from chemprop.cli.predict import PredictSubcommand
from chemprop.cli.convert import ConvertSubcommand

# TODO: add subcommands for Fingerprint and Hyperopt
# from chemprop.cli.fingerprint import FingerprintSubcommand
# from chemprop.cli.hyperopt import HyperoptSubcommand

from chemprop.cli.utils import NOW, pop_attr

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
    parent.add_argument("--output-dir", "--save-dir", help="Directory where outputs will be saved.")
    parent.add_argument("-v", "--verbose", action="count", default=0, help="the verbosity level")

    parents = [parent]
    for subcommand in SUBCOMMANDS:
        subcommand.add(subparsers, parents)

    args = parser.parse_args()
    verbose, mode, func = (
        pop_attr(args, attr) for attr in ["verbose", "mode", "func"]
    )

    args.output_dir = (
        Path.cwd() / "output_files" / mode if args.output_dir is None else args.output_dir
    ) / NOW
    args.output_dir.mkdir(parents=True, exist_ok=True)
    logfile = args.output_dir / "chemprop.log"

    logging.basicConfig(
        filename=str(logfile),
        format="%(asctime)s - %(levelname)s:%(name)s - %(message)s",
        level=LOG_LEVELS[min(verbose, len(LOG_LEVELS) - 1)],
        datefmt="%Y-%m-%dT%H:%M:%S",
        force=True,
    )

    logger.info(f"Running in mode '{mode}' with args: {vars(args)}")

    func(args)
