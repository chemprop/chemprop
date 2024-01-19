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
    parent.add_argument(
        "--output-dir",
        "--save-dir",
        type=Path,
        help="Directory where outputs will be saved. Defaults to <current working directory>/<mode>/<date time>.",
    )
    parent.add_argument(
        "-q", "--quiet", action="count", default=0, help="supression level for logging"
    )

    parents = [parent]
    for subcommand in SUBCOMMANDS:
        subcommand.add(subparsers, parents)

    args = parser.parse_args()
    quiet, mode, func = (pop_attr(args, attr) for attr in ["quiet", "mode", "func"])

    args.output_dir = (args.output_dir or Path.cwd() / "output_files" / mode) / NOW
    args.output_dir.mkdir(parents=True, exist_ok=True)
    logfile = args.output_dir / "chemprop.log"

    logging.basicConfig(
        filename=str(logfile),
        format="%(asctime)s - %(levelname)s:%(name)s - %(message)s",
        level=LOG_LEVELS[max(0, len(LOG_LEVELS) - 1 - quiet)],
        datefmt="%Y-%m-%dT%H:%M:%S",
        force=True,
    )

    logger.info(f"Running in mode '{mode}' with args: {vars(args)}")

    func(args)
