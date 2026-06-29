from lightning.pytorch.callbacks import Callback

from chemprop.utils.registry import ClassRegistry

from .interpret import MyersonExplainerCallback

CallbackRegistry = ClassRegistry[Callback]()


__all__ = ["CallbackRegistry", "MyersonExplainerCallback"]
