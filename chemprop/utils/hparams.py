from typing import Protocol, Type, TypedDict


class HParamsDict(TypedDict):
    """A dictionary containing a module's class and it's hyperparameters

    Using this type should essentially allow for initializing a module via::

        module = hparams.pop('cls')(**hparams)
    """

    cls: Type


class HasHParams(Protocol):
    """:class:`HasHParams` is a protocol for clases which possess an :attr:`hparams` attribute which is a dictionary containing the object's class and arguments required to initialize it.

    That is, any object which implements :class:`HasHParams` should be able to be initialized via::

        class Foo(HasHParams):
            def __init__(self, *args, **kwargs):
                ...

        foo1 = Foo(...)
        foo1_cls = foo1.hparams['cls']
        foo1_kwargs = {k: v for k, v in foo1.hparams.items() if k != "cls"}
        foo2 = foo1_cls(**foo1_kwargs)
        # code to compare foo1 and foo2 goes here and they should be equal
    """

    hparams: HParamsDict


def from_hparams(hparams: HParamsDict):
    cls = hparams["cls"]
    kwargs = {k: v for k, v in hparams.items() if k != "cls"}

    return cls(**kwargs)
