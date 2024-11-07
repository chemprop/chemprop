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


  
