from typing import Iterable

from chemprop.utils import pretty_shape


class InvalidShapeError(ValueError):
    def __init__(self, var_name: str, received: Iterable[int], expected: Iterable[int]):
        message = (
            f"arg '{var_name}' has incorrect shape! "
            f"got: `{pretty_shape(received)}`. expected: `{pretty_shape(expected)}`"
        )
        super().__init__(message)
