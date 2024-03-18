import logging
from argparse import ArgumentParser, Namespace

from ray.train.lightning import (RayDDPStrategy, RayLightningEnvironment,
                                 RayTrainReportCallback, prepare_trainer)

from chemprop.cli.utils.command import Subcommand

logger = logging.getLogger(__name__)


class HyperoptSubcommand(Subcommand):
    COMMAND = "hyperopt"
    HELP = "perform hyperparameter optimization on the given task"

    @classmethod
    def add_args(cls, parser: ArgumentParser) -> ArgumentParser:
        pass

    @classmethod
    def func(cls, args: Namespace):
        pass

def main(args: Namespace):
    pass
