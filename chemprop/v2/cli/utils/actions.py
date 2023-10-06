from argparse import Action, ArgumentParser, Namespace
from typing import Any, Mapping, Sequence


def LookupAction(obj: Mapping[str, Any]):
    class LookupAction_(Action):
        def __init__(self, option_strings, dest, default=None, choices=None, **kwargs):
            if default not in obj.keys() and default is not None:
                raise ValueError(
                    f"Invalid value for arg 'default': '{default}'. "
                    f"Expected one of {tuple(obj.keys())}"
                )

            kwargs["choices"] = choices if choices is not None else obj.keys()
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

    return LookupAction_
