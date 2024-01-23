from argparse import ArgumentParser, Namespace
import sys
import logging
from pathlib import Path

from chemprop.cli.utils import Subcommand
from chemprop.utils.v1_to_v2 import convert_model_file_v1_to_v2

logger = logging.getLogger(__name__)


class ConvertSubcommand(Subcommand):
    COMMAND = "convert"
    HELP = "convert a v1 model checkpoint (.pt) to a v2 model checkpoint (.ckpt)"

    @classmethod
    def add_args(cls, parser: ArgumentParser) -> ArgumentParser:
        parser.add_argument(
            "-i",
            "--input-path",
            required=True,
            type=Path,
            help="The path to a v1 model .pt checkpoint file.",
        )
        parser.add_argument(
            "-o",
            "--output-path",
            type=Path,
            help="The path to which the converted model will be saved. Defaults to '<current working directory>/<stem of input>_v2.ckpt'",
        )
        return parser

    @classmethod
    def func(cls, args: Namespace):
        args.output_path = args.output_path or Path(args.input_path.stem + "_v2.ckpt")
        if args.output_path.suffix != ".ckpt":
            args.output_path = Path(str(args.output_path) + ".ckpt")

        logger.info(
            f"Converting v1 model checkpoint '{args.input_path}' to v2 model checkpoint '{args.output_path}'..."
        )
        convert_model_file_v1_to_v2(args.input_path, args.output_path)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser = ConvertSubcommand.add_args(parser)

    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG, force=True)

    args = parser.parse_args()
    ConvertSubcommand.func(args)
