from lightning.pytorch.callbacks import Callback

from chemprop.utils.registry import ClassRegistry

CallbackRegistry = ClassRegistry[Callback]()

from .example_callbacks import ExampleCallback
from .interpret_callbacks import MyersonExplainerCallback

__all__ = ["CallbackRegistry", "ExampleCallback", "MyersonExplainerCallback"]
