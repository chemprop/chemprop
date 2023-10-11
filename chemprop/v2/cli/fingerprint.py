from argparse import ArgumentParser, Namespace
from enum import auto

from chemprop.v2.cli.utils.command import Subcommand
from chemprop.v2.utils.utils import EnumMapping


class RepresentationType(EnumMapping):
    FINGERPRINT = auto()
    ENCODING = auto()


class FingerprintSubcommand(Subcommand):
    COMMAND = "fingerprint"
    HELP = "use a pretrained chemprop model for to calculate learned representations"

    @classmethod
    def add_args(cls, parser: ArgumentParser) -> ArgumentParser:
        parser.add_argument(
            "--repr-type", type=RepresentationType.get, choices=RepresentationType.keys()
        )

    @classmethod
    def func(cls, args: Namespace):
        pass
