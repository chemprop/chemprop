from lightning.pytorch.callbacks import Callback

from chemprop.callbacks import CallbackRegistry

@CallbackRegistry.register("example_callback")
class ExampleCallback(Callback):
    def __init__(self, cli_args):
        super().__init__()

    def on_predict_epoch_start(self, trainer, pl_module):
        pass

    def on_predict_epoch_end(self, trainer, pl_module):
        pass

    def on_predict_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx=0):
        pass

    def on_predict_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        pass

    def on_predict(self, trainer, pl_module):
        pass

    def on_predict(self, trainer, pl_module):
        pass