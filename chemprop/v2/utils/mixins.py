import inspect
from typing import Any, Collection, Iterable


class ReprMixin:
    def __repr__(self) -> str:
        items = self.get_params()
        
        if len(items) > 0:
            keys, values = zip(*items)
            sig = inspect.signature(self.__class__)
            defaults = [sig.parameters[k].default for k in keys]
            items = [(k, v) for k, v, d in zip(keys, values, defaults) if v != d]

        argspec = ", ".join(f"{k}={repr(v)}" for k, v in items)

        return f"{self.__class__.__name__}({argspec})"

    def get_params(self) -> Collection[tuple[str, Any]]:
        return self.__dict__.items()


class RegistryMixin:
    """The `RegistryMixin` class automatically registers all subclasses into a class-level
    registry.

    Notes
    -----
    1. classes that utilize this mixin must define a `registry` attribute
    2. each subclass that is to be registered must define an `alias` class variable. Omitting this
        will result in the class to be ommitted from the registry. NOTE: this is useful in the case
        of intermediate ABCs. Ex:
                A (parent ABC with registry)
                |
                v
                B (subclass ABC that defines some common methods for concrete child classes)
                |
                v
          C_1, C_2, C_3 (subclasses to be registered)
    """

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if hasattr(cls, "alias"):
            cls.registry[cls.alias] = cls


class FactoryMixin:
    @classmethod
    def build(cls, alias: str, *args, **kwargs):
        try:
            return cls.registry[alias](*args, **kwargs)
        except KeyError:
            raise ValueError(
                f"Invalid {cls.__name__}! got: '{alias}', expected one of: {cls.choices}"
            )
    
    @classmethod
    @property
    def choices(cls) -> set[str]:
        return set(cls.registry.keys())
