from lightning.pytorch.callbacks import Callback
from chemprop.utils.registry import ClassRegistry
from .example_callbacks import ExampleCallback
from .interpret_callbacks import MyersonExplainerCallback

CallbackRegistry = ClassRegistry[Callback]()

__all__ = ["CallbackRegistry", "ExampleCallback", "MyersonExplainerCallback"]
