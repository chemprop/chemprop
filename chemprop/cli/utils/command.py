from abc import ABC, abstractmethod
from argparse import ArgumentParser, Namespace, _SubParsersAction


class Subcommand(ABC):
    COMMAND: str
    HELP: str | None = None

    @classmethod
    def add(cls, subparsers: _SubParsersAction, parents) -> ArgumentParser:
        parser = subparsers.add_parser(cls.COMMAND, help=cls.HELP, parents=parents)
        cls.add_args(parser).set_defaults(func=cls.func)

        return parser

    @classmethod
    @abstractmethod
    def add_args(cls, parser: ArgumentParser) -> ArgumentParser:
        pass

    @classmethod
    @abstractmethod
    def func(cls, args: Namespace):
        pass
