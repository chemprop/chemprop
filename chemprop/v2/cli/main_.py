from argparse import ArgumentParser
from datetime import datetime
import logging

import yaml

from chemprop.v2.cli import train

logger = logging.getLogger(__name__)


def main():
    parser = ArgumentParser()
    subparsers = parser.add_subparsers(title="mode", dest="mode")

    parser_train = subparsers.add_parser("train")
    train.add_args(parser_train)
    parser_train.set_defaults(run=train.main, fix=train.process_args)

    parser_predict = subparsers.add_parser("predict")

    parser_fingerprint = subparsers.add_parser("fingerprint")

    parser_interpret = subparsers.add_parser("interpret")

    args, _ = parser.parse_known_args()
    params = parser_train.get_items_for_config_file_output(parser_train._source_to_settings, args)

    args.fix(args)

    start = datetime.now()
    s_now = start.strftime("%Y-%m-%dT%H:%M:%S.log")
    logging.basicConfig(
        filename=args.logdir / s_now,
        format="%(asctime)s - %(levelname)s - %(name)s : %(message)s",
        level=logging.DEBUG,
        force=True,
    )

    logger.info(f"Starting at {start.ctime()}")
    logger.info(f"Running in mode: {args.mode}")
    logger.info(parser.format_values().strip())
    logger.info(f"Running with params: {dict(params)}")

    p_config = args.output_dir / "config.yaml"
    if p_config.exists():
        logger.info(f"Overwriting config file at {p_config}")
    else:
        logger.info(f"Writing config file to {p_config}")
    p_config.write_text(yaml.dump(dict(params), indent=2))

    args.run(args)
    end = datetime.now()
    logger.info(f"Finished at {end.ctime()}")

    elapsed = end - start
    logger.info(f"Total time: {elapsed}")


if __name__ == "__main__":
    main()
