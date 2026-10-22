from argparse import ArgumentError, ArgumentParser, Namespace
import logging
from pathlib import Path
import sys

from chemprop.cli.utils import Subcommand
from chemprop.utils.v1_to_v2 import convert_model_file_v1_to_v2
from chemprop.utils.v2_0_to_v2_1 import convert_model_file_v2_0_to_v2_1
from chemprop.utils.v2_1_to_v2_4 import (
    convert_model_file_to_latest,
    convert_model_file_v2_1_to_v2_4,
)

logger = logging.getLogger(__name__)


_SUFFIXES = {
    "v1_to_v2": "_v2",
    "v2_0_to_v2_1": "_v2_1",
    "v2_1_to_v2_4": "_v2_4",
    "latest": "_latest",
}
# these conversions load the input file with `weights_only=False`, so ask if it is trusted
_PICKLE_CONVERSIONS = ("v2_1_to_v2_4", "latest")


class ConvertSubcommand(Subcommand):
    COMMAND = "convert"
    HELP = "Convert model checkpoint (.pt or .ckpt) to more recent version."

    @classmethod
    def add_args(cls, parser: ArgumentParser) -> ArgumentParser:
        parser.add_argument(
            "-c",
            "--conversion",
            choices=list(_SUFFIXES.keys()),
            help="Conversion to perform. ``latest`` converts a model file from any version to the latest format, which can be loaded with ``--weights-only true``, by applying the other conversions as needed. ``v2_1_to_v2_4`` converts a model file from v2.1 or later to the same format. Models converted from v1 must be run with the v1 featurizer via ``--multi-hot-atom-featurizer-mode v1``.",
            default="latest",
        )
        parser.add_argument(
            "-i",
            "--input-path",
            required=True,
            type=Path,
            help="Path to a model .pt or checkpoint .ckpt file",
        )
        parser.add_argument(
            "-o",
            "--output-path",
            type=Path,
            help="Path to which the converted model will be saved (``CURRENT_DIRECTORY/STEM_OF_INPUT_newversion.pt`` by default)",
        )
        parser.add_argument(
            "-y",
            "--yes",
            action="store_true",
            help="Confirm that the input model file is trusted without prompting. The ``latest`` and ``v2_1_to_v2_4`` conversions load the input file with ``weights_only=False``, which can execute arbitrary code.",
        )
        return parser

    @classmethod
    def func(cls, args: Namespace):
        if args.output_path is None:
            suffix = args.input_path.suffix if args.input_path.suffix == ".ckpt" else ".pt"
            args.output_path = Path(args.input_path.stem + _SUFFIXES[args.conversion] + suffix)
        if args.output_path.suffix not in (".pt", ".ckpt"):
            raise ArgumentError(
                argument=None,
                message=f"Output must be a `.pt` or `.ckpt` file. Got {args.output_path}",
            )

        if args.conversion in _PICKLE_CONVERSIONS and not confirm_trusted(
            args.input_path, args.yes
        ):
            logger.info("Conversion cancelled.")
            return

        logger.info(
            f"Converting model checkpoint '{args.input_path}' to model checkpoint '{args.output_path}'..."
        )
        match args.conversion:
            case "v1_to_v2":
                convert_model_file_v1_to_v2(args.input_path, args.output_path)
            case "v2_0_to_v2_1":
                convert_model_file_v2_0_to_v2_1(args.input_path, args.output_path)
            case "v2_1_to_v2_4":
                convert_model_file_v2_1_to_v2_4(args.input_path, args.output_path)
            case "latest":
                convert_model_file_to_latest(args.input_path, args.output_path)


def confirm_trusted(input_path: Path, yes: bool) -> bool:
    """Ask the user to confirm that the model file is trusted before loading it with
    ``weights_only=False``"""
    logger.warning(
        f"Converting '{input_path}' requires loading it with `weights_only=False`, which can "
        "execute arbitrary code. Only convert model files you trust."
    )
    if yes:
        return True
    if not sys.stdin.isatty():
        raise ArgumentError(
            argument=None,
            message="Cannot ask whether the model file is trusted in a non-interactive session. "
            "Pass `--yes` to confirm that it is trusted.",
        )

    answer = input(f"Do you trust the model file '{input_path}'? [y/N]: ")
    return answer.strip().lower() in ("y", "yes")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser = ConvertSubcommand.add_args(parser)

    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG, force=True)

    args = parser.parse_args()
    ConvertSubcommand.func(args)
