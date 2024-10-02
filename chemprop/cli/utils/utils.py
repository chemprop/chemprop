from typing import Any

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
