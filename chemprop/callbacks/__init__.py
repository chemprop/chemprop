from lightning.pytorch.callbacks import Callback

from chemprop.utils.registry import ClassRegistry

CallbackRegistry = ClassRegistry[Callback]()

from .interpret import MyersonExplainerCallback  # noqa: E402 # avoid circular import

__all__ = ["CallbackRegistry", "MyersonExplainerCallback"]
