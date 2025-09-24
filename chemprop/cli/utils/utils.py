from typing import Any

import numpy as np

__all__ = ["pop_attr"]


def pop_attr(o: object, attr: str, *args) -> Any | None:
    """like ``pop()`` but for attribute maps"""
    match len(args):
        case 0:
            return _pop_attr(o, attr)
        case 1:
            return _pop_attr_d(o, attr, args[0])
        case _:
            raise TypeError(f"Expected at most 2 arguments! got: {len(args)}")


def _pop_attr(o: object, attr: str) -> Any:
    val = getattr(o, attr)
    delattr(o, attr)

    return val


def _pop_attr_d(o: object, attr: str, default: Any | None = None) -> Any | None:
    try:
        val = getattr(o, attr)
        delattr(o, attr)
    except AttributeError:
        val = default

    return val


def _to_str(number: float) -> str:
    return f"{number:.6e}"


def format_probability_string(test_preds: np.ndarray) -> np.ndarray:
    axis = test_preds.ndim - 1
    formatted_probability_strings = np.apply_along_axis(
        lambda x: ",".join(map(_to_str, x)), axis, test_preds
    )
    return formatted_probability_strings
