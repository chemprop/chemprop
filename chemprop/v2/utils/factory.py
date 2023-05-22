import inspect

from chemprop.v2.utils.registry import ClassRegistry


class ClassFactory(ClassRegistry):   
    def build(self, key: str, **kwargs) -> object:
        try:
            clz = self[key.lower()]
        except KeyError:
            raise ValueError(f"'{key}' is not a valid key! Expected one of: {set(self.keys())}.")

        sig = inspect.signature(clz)
        kwargs = {k: v for k, v in kwargs.items() if k in sig.parameters.keys()}

        return clz(**kwargs)
