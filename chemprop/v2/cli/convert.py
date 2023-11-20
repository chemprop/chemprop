import sys
import logging
from pathlib import Path
from argparse import ArgumentParser, Namespace

from chemprop.v2.cli.utils import Subcommand
from chemprop.v2.utils.v1_to_v2 import convert_model_file_v1_to_v2

logger = logging.getLogger(__name__)


class ConvertSubcommand(Subcommand):
    COMMAND = "convert"
    HELP = "convert a v1 model checkpoint (.pt) to a v2 model checkpoint (.ckpt)"

    @classmethod
    def add_args(cls, parser: ArgumentParser) -> ArgumentParser:
        parser.add_argument("input_path", help="The path to a v1 model .pt checkpoint file.")
        parser.add_argument(
            "output_path",
            nargs="?",
            help="The path to which the converted model will be saved. If not specified and the input file is '/path/to/checkpoint/model.pt', the output will default to '/path/to/checkpoint/model_v2.ckpt'",
        )
        return parser

    @classmethod
    def func(cls, args: Namespace):
        args.input_path = Path(args.input_path)
        args.output_path = Path(
            args.output_path or args.input_path.parent / (args.input_path.stem + "_v2.ckpt")
        )

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
