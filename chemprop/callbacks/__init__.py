from lightning.pytorch.callbacks import Callback
from chemprop.utils.registry import ClassRegistry

CallbackRegistry = ClassRegistry[Callback]()

from .example_callbacks import ExampleCallback

__all__ = ["CallbackRegistry", "ExampleCallback"]
