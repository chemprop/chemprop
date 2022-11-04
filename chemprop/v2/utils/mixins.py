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
