from argparse import ArgumentParser, Namespace

from chemprop.cli.utils.command import Subcommand


class HyperoptSubcommand(Subcommand):
    COMMAND = "hyperopt"
    HELP = "perform hyperparameter optimization on the given task"

    @classmethod
    def add_args(cls, parser: ArgumentParser) -> ArgumentParser:
        pass

    @classmethod
    def func(cls, args: Namespace):
        pass
