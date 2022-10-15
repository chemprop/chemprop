from __future__ import annotations

from enum import Enum
from typing import Union


class AutoName(Enum):
    def _generate_next_value_(name, start, count, last_values):
        return name

    @classmethod
    def get(cls, name: Union[str, AutoName]) -> AutoName:
        if isinstance(name, cls):
            return name

        try:
            return cls[name.upper()]
        except KeyError:
            names = [x.value for x in cls]
            raise ValueError(f"Invalid name! got: '{name}'. expected one of: {tuple(names)}")
