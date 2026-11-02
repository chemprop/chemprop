from argparse import ArgumentError, ArgumentParser, Namespace
import logging
from pathlib import Path
import pickle
import sys

from chemprop.cli.utils import Subcommand
from chemprop.cli.utils.args import bool_arg
from chemprop.utils.convert_to_latest import convert_model_file_to_latest
from chemprop.utils.v1_to_v2 import convert_model_file_v1_to_v2
from chemprop.utils.v2_0_to_v2_1 import convert_model_file_v2_0_to_v2_1
from chemprop.utils.v2_1_to_v2_4 import convert_model_file_v2_1_to_v2_4

logger = logging.getLogger(__name__)


_SUFFIXES = {
    "v1_to_v2": "_v2",
    "v2_0_to_v2_1": "_v2_1",
    "v2_1_to_v2_4": "_v2_4",
    "latest": "_latest",
}
_CONVERSIONS = {
    "v1_to_v2": convert_model_file_v1_to_v2,
    "v2_0_to_v2_1": convert_model_file_v2_0_to_v2_1,
    "v2_1_to_v2_4": convert_model_file_v2_1_to_v2_4,
    "latest": convert_model_file_to_latest,
}


class ConvertSubcommand(Subcommand):
    COMMAND = "convert"
    HELP = "Convert model checkpoint (.pt or .ckpt) to more recent version."

    @classmethod
    def add_args(cls, parser: ArgumentParser) -> ArgumentParser:
        parser.add_argument(
            "-c",
            "--conversion",
            choices=list(_SUFFIXES.keys()),
            help="Conversion to perform. Models converted from v1 to v2 must be run with the v1 featurizer via ``--multi-hot-atom-featurizer-mode v1``.",
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
            "--weights-only",
            type=bool_arg,
            help="Passed as ``weights_only`` to ``torch.load`` when loading the input model file (``true`` or ``false``). Know the risks before passing ``false``. Model files saved before Chemprop v2.4 can only be loaded with ``false``. If not specified, ``false`` is used with a warning that the default will change to ``true`` in a future release.",
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

        logger.info(
            f"Converting model checkpoint '{args.input_path}' to model checkpoint '{args.output_path}'..."
        )
        convert = _CONVERSIONS[args.conversion]
        try:
            convert(args.input_path, args.output_path, weights_only=args.weights_only)
        except (pickle.UnpicklingError, RuntimeError) as e:
            if not args.weights_only:
                raise
            raise pickle.UnpicklingError(
                f"Could not convert '{args.input_path}' with `--weights-only true`. Converting a "
                "model file saved before Chemprop v2.4 requires `--weights-only false`, which can "
                "execute arbitrary code. If you trust this file, pass `--weights-only false`."
            ) from e


if __name__ == "__main__":
    parser = ArgumentParser()
    parser = ConvertSubcommand.add_args(parser)

    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG, force=True)

    args = parser.parse_args()
    ConvertSubcommand.func(args)
