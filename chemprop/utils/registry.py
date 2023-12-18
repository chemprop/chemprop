import inspect
from typing import Any, Iterable, Type, TypeVar

T = TypeVar("T")


class ClassRegistry(dict[str, Type[T]]):
    def register(self, alias: Any | Iterable[Any] | None = None):
        def decorator(cls):
            if alias is None:
                keys = [cls.__name__.lower()]
            elif isinstance(alias, str):
                keys = [alias]
            else:
                keys = alias

            cls.alias = keys[0]
            for k in keys:
                self[k] = cls

            return cls

        return decorator

    __call__ = register

    def __repr__(self) -> str:  # pragma: no cover
        return f"{self.__class__.__name__}: {super().__repr__()}"

    def __str__(self) -> str:  # pragma: no cover
        INDENT = 4
        items = [f"{' ' * INDENT}{repr(k)}: {repr(v)}" for k, v in self.items()]

        return "\n".join([f"{self.__class__.__name__} {'{'}", ",\n".join(items), "}"])


class Factory:
    @classmethod
    def build(cls, clz_T: Type[T], *args, **kwargs) -> T:
        if not inspect.isclass(clz_T):
            raise TypeError(f"Expected a class type! got: {type(clz_T)}")

        sig = inspect.signature(clz_T)
        kwargs = {k: v for k, v in kwargs.items() if k in sig.parameters.keys()}

        return clz_T(*args, **kwargs)
