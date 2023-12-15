import functools

__all__ = ["bounded"]


def bounded(lo: float | None = None, hi: float | None = None):
    if lo is None and hi is None:
        raise ValueError("No bounds provided!")

    def decorator(f):
        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            x = f(*args, **kwargs)

            if (lo is not None and hi is not None) and not lo <= x <= hi:
                raise ValueError(f"Parsed value outside of range [{lo}, {hi}]! got: {x}")
            if hi is not None and x > hi:
                raise ValueError(f"Parsed value below {hi}! got: {x}")
            if lo is not None and x < lo:
                raise ValueError(f"Parsed value above {lo}]! got: {x}")

            return x

        return wrapper

    return decorator


def uppercase(x: str):
    return x.upper()


def lowercase(x: str):
    return x.lower()
