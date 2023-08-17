from argparse import Action, ArgumentParser, Namespace
from typing import Any, Sequence

from chemprop.v2.utils import ClassRegistry

def RegistryAction(cls: ClassRegistry):
    class RegistryAction_(Action):
        def __init__(self, option_strings, dest, default=None, choices=None, **kwargs):
            if default not in cls.keys() and default is not None:
                raise ValueError(
                    f"Invalid value for arg 'default': '{default}'. "
                    f"Expected one of {tuple(cls.keys())}"
                )

            kwargs["choices"] = choices if choices is not None else cls.keys()
            kwargs["default"] = default

            super().__init__(option_strings, dest, **kwargs)

        def __call__(
            self,
            parser: ArgumentParser,
            namespace: Namespace,
            values: str | Sequence[Any] | None,
            option_string: str | None = None,
        ):
            setattr(namespace, self.dest, values)

    return RegistryAction_
