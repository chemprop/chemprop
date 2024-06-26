from argparse import _StoreAction
from typing import Any, Mapping


def LookupAction(obj: Mapping[str, Any]):
    class LookupAction_(_StoreAction):
        def __init__(self, option_strings, dest, default=None, choices=None, **kwargs):
            if default not in obj.keys() and default is not None:
                raise ValueError(
                    f"Invalid value for arg 'default': '{default}'. "
                    f"Expected one of {tuple(obj.keys())}"
                )

            kwargs["choices"] = choices if choices is not None else obj.keys()
            kwargs["default"] = default

            super().__init__(option_strings, dest, **kwargs)

    return LookupAction_
