:py:mod:`models.model`
======================

.. py:module:: models.model


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   models.model.MPNN




.. py:class:: MPNN(message_passing, agg, readout, batch_norm = True, metrics = None, w_t = None, warmup_epochs = 2, init_lr = 0.0001, max_lr = 0.001, final_lr = 0.0001)


   Bases: :py:obj:`lightning.pytorch.LightningModule`

   An :class:`MPNN` is a sequence of message passing layers, an aggregation routine, and a
   readout routine.

   The first two modules calculate learned fingerprints from an input molecule
   reaction graph, and the final module takes these leared fingerprints as input to calculate a
   final prediction. I.e., the following operation:

   .. math::
       \mathtt{MPNN}(\mathcal{G}) =
           \mathtt{readout}(\mathtt{agg}(\mathtt{message\_passing}(\mathcal{G})))

   The full model is trained end-to-end.

   :param message_passing: the message passing block to use to calculate learned fingerprints
   :type message_passing: MessagePassingBlock
   :param agg: the aggregation operation to use during molecule-level readout
   :type agg: Aggregation
   :param readout: the readout operation to use to calculate the final prediction
   :type readout: Readout
   :param batch_norm: if `True`, apply batch normalization to the output of the aggregation operation
   :type batch_norm: bool, default=True
   :param metrics: the metrics to use to evaluate the model during training and evaluation
   :type metrics: Iterable[Metric] | None, default=None
   :param w_t: the weights to use for each task during training. If `None`, use uniform weights
   :type w_t: Tensor | None, default=None
   :param warmup_epochs: the number of epochs to use for the learning rate warmup
   :type warmup_epochs: int, default=2
   :param # num_lrs:
   :type # num_lrs: int, default=1
   :param #     the number of learning rates to use during training:
   :param init_lr: the initial learning rate
   :type init_lr: int, default=1e-4
   :param max_lr: the maximum learning rate
   :type max_lr: float, default=1e-3
   :param final_lr: the final learning rate
   :type final_lr: float, default=1e-4

   :raises ValueError: if the output dimension of the message passing block does not match the input dimension of
       the readout block

   .. py:property:: output_dim
      :type: int


   .. py:property:: n_tasks
      :type: int


   .. py:property:: n_targets
      :type: int


   .. py:property:: criterion
      :type: chemprop.v2.models.loss.LossFunction


   .. py:method:: fingerprint(bmg, V_d = None, X_f = None)

      the learned fingerprints for the input molecules


   .. py:method:: encoding(bmg, V_d = None, X_f = None)

      the final hidden representations for the input molecules


   .. py:method:: forward(bmg, V_d = None, X_f = None)

      Generate predictions for the input molecules/reactions


   .. py:method:: training_step(batch, batch_idx)

      Here you compute and return the training loss and some additional metrics for e.g.
      the progress bar or logger.

      :param batch: The output of your :class:`~torch.utils.data.DataLoader`. A tensor, tuple or list.
      :type batch: :class:`~torch.Tensor` | (:class:`~torch.Tensor`, ...) | [:class:`~torch.Tensor`, ...]
      :param batch_idx: Integer displaying index of this batch
      :type batch_idx: ``int``

      :returns: Any of.

                - :class:`~torch.Tensor` - The loss tensor
                - ``dict`` - A dictionary. Can include any keys, but must include the key ``'loss'``
                - ``None`` - Training will skip to the next batch. This is only for automatic optimization.
                    This is not supported for multi-GPU, TPU, IPU, or DeepSpeed.

      In this step you'd normally do the forward pass and calculate the loss for a batch.
      You can also do fancier things like multiple forward passes or something model specific.

      Example::

          def training_step(self, batch, batch_idx):
              x, y, z = batch
              out = self.encoder(x)
              loss = self.loss(out, x)
              return loss

      To use multiple optimizers, you can switch to 'manual optimization' and control their stepping:

      .. code-block:: python

          def __init__(self):
              super().__init__()
              self.automatic_optimization = False


          # Multiple optimizers (e.g.: GANs)
          def training_step(self, batch, batch_idx):
              opt1, opt2 = self.optimizers()

              # do training_step with encoder
              ...
              opt1.step()
              # do training_step with decoder
              ...
              opt2.step()

      .. note::

         When ``accumulate_grad_batches`` > 1, the loss returned here will be automatically
         normalized by ``accumulate_grad_batches`` internally.


   .. py:method:: validation_step(batch, batch_idx = 0)

      Operates on a single batch of data from the validation set.
      In this step you'd might generate examples or calculate anything of interest like accuracy.

      :param batch: The output of your :class:`~torch.utils.data.DataLoader`.
      :param batch_idx: The index of this batch.
      :param dataloader_idx: The index of the dataloader that produced this batch.
                             (only if multiple val dataloaders used)

      :returns:

                - Any object or value
                - ``None`` - Validation will skip to the next batch

      .. code-block:: python

          # if you have one val dataloader:
          def validation_step(self, batch, batch_idx):
              ...


          # if you have multiple val dataloaders:
          def validation_step(self, batch, batch_idx, dataloader_idx=0):
              ...

      Examples::

          # CASE 1: A single validation dataset
          def validation_step(self, batch, batch_idx):
              x, y = batch

              # implement your own
              out = self(x)
              loss = self.loss(out, y)

              # log 6 example images
              # or generated text... or whatever
              sample_imgs = x[:6]
              grid = torchvision.utils.make_grid(sample_imgs)
              self.logger.experiment.add_image('example_images', grid, 0)

              # calculate acc
              labels_hat = torch.argmax(out, dim=1)
              val_acc = torch.sum(y == labels_hat).item() / (len(y) * 1.0)

              # log the outputs!
              self.log_dict({'val_loss': loss, 'val_acc': val_acc})

      If you pass in multiple val dataloaders, :meth:`validation_step` will have an additional argument. We recommend
      setting the default value of 0 so that you can quickly switch between single and multiple dataloaders.

      .. code-block:: python

          # CASE 2: multiple validation dataloaders
          def validation_step(self, batch, batch_idx, dataloader_idx=0):
              # dataloader_idx tells you which dataset this is.
              ...

      .. note:: If you don't need to validate you don't need to implement this method.

      .. note::

         When the :meth:`validation_step` is called, the model has been put in eval mode
         and PyTorch gradients have been disabled. At the end of validation,
         the model goes back to training mode and gradients are enabled.


   .. py:method:: test_step(batch, batch_idx = 0)

      Operates on a single batch of data from the test set.
      In this step you'd normally generate examples or calculate anything of interest
      such as accuracy.

      :param batch: The output of your :class:`~torch.utils.data.DataLoader`.
      :param batch_idx: The index of this batch.
      :param dataloader_id: The index of the dataloader that produced this batch.
                            (only if multiple test dataloaders used).

      :returns:

                Any of.

                 - Any object or value
                 - ``None`` - Testing will skip to the next batch

      .. code-block:: python

          # if you have one test dataloader:
          def test_step(self, batch, batch_idx):
              ...


          # if you have multiple test dataloaders:
          def test_step(self, batch, batch_idx, dataloader_idx=0):
              ...

      Examples::

          # CASE 1: A single test dataset
          def test_step(self, batch, batch_idx):
              x, y = batch

              # implement your own
              out = self(x)
              loss = self.loss(out, y)

              # log 6 example images
              # or generated text... or whatever
              sample_imgs = x[:6]
              grid = torchvision.utils.make_grid(sample_imgs)
              self.logger.experiment.add_image('example_images', grid, 0)

              # calculate acc
              labels_hat = torch.argmax(out, dim=1)
              test_acc = torch.sum(y == labels_hat).item() / (len(y) * 1.0)

              # log the outputs!
              self.log_dict({'test_loss': loss, 'test_acc': test_acc})

      If you pass in multiple test dataloaders, :meth:`test_step` will have an additional argument. We recommend
      setting the default value of 0 so that you can quickly switch between single and multiple dataloaders.

      .. code-block:: python

          # CASE 2: multiple test dataloaders
          def test_step(self, batch, batch_idx, dataloader_idx=0):
              # dataloader_idx tells you which dataset this is.
              ...

      .. note:: If you don't need to test you don't need to implement this method.

      .. note::

         When the :meth:`test_step` is called, the model has been put in eval mode and
         PyTorch gradients have been disabled. At the end of the test epoch, the model goes back
         to training mode and gradients are enabled.


   .. py:method:: predict_step(batch, batch_idx, dataloader_idx = 0)

      Return the predictions of the input batch

      :param batch: the input batch
      :type batch: TrainingBatch

      :returns: a tensor of varying shape depending on the task type:

                * regression/binary classification: ``n x (t * s)``, where ``n`` is the number of input
                molecules/reactions, ``t`` is the number of tasks, and ``s`` is the number of targets
                per task. The final dimension is flattened, so that the targets for each task are
                grouped. I.e., the first ``t`` elements are the first target for each task, the second
                ``t`` elements the second target, etc.
                * multiclass classification: ``n x t x c``, where ``c`` is the number of classes
      :rtype: Tensor


   .. py:method:: configure_optimizers()

      Choose what optimizers and learning-rate schedulers to use in your optimization.
      Normally you'd need one. But in the case of GANs or similar you might have multiple.
      Optimization with multiple optimizers only works in the manual optimization mode.

      :returns: Any of these 6 options.

                - **Single optimizer**.
                - **List or Tuple** of optimizers.
                - **Two lists** - The first list has multiple optimizers, and the second has multiple LR schedulers
                  (or multiple ``lr_scheduler_config``).
                - **Dictionary**, with an ``"optimizer"`` key, and (optionally) a ``"lr_scheduler"``
                  key whose value is a single LR scheduler or ``lr_scheduler_config``.
                - **None** - Fit will run without any optimizer.

      The ``lr_scheduler_config`` is a dictionary which contains the scheduler and its associated configuration.
      The default configuration is shown below.

      .. code-block:: python

          lr_scheduler_config = {
              # REQUIRED: The scheduler instance
              "scheduler": lr_scheduler,
              # The unit of the scheduler's step size, could also be 'step'.
              # 'epoch' updates the scheduler on epoch end whereas 'step'
              # updates it after a optimizer update.
              "interval": "epoch",
              # How many epochs/steps should pass between calls to
              # `scheduler.step()`. 1 corresponds to updating the learning
              # rate after every epoch/step.
              "frequency": 1,
              # Metric to to monitor for schedulers like `ReduceLROnPlateau`
              "monitor": "val_loss",
              # If set to `True`, will enforce that the value specified 'monitor'
              # is available when the scheduler is updated, thus stopping
              # training if not found. If set to `False`, it will only produce a warning
              "strict": True,
              # If using the `LearningRateMonitor` callback to monitor the
              # learning rate progress, this keyword can be used to specify
              # a custom logged name
              "name": None,
          }

      When there are schedulers in which the ``.step()`` method is conditioned on a value, such as the
      :class:`torch.optim.lr_scheduler.ReduceLROnPlateau` scheduler, Lightning requires that the
      ``lr_scheduler_config`` contains the keyword ``"monitor"`` set to the metric name that the scheduler
      should be conditioned on.

      .. testcode::

          # The ReduceLROnPlateau scheduler requires a monitor
          def configure_optimizers(self):
              optimizer = Adam(...)
              return {
                  "optimizer": optimizer,
                  "lr_scheduler": {
                      "scheduler": ReduceLROnPlateau(optimizer, ...),
                      "monitor": "metric_to_track",
                      "frequency": "indicates how often the metric is updated"
                      # If "monitor" references validation metrics, then "frequency" should be set to a
                      # multiple of "trainer.check_val_every_n_epoch".
                  },
              }


          # In the case of two optimizers, only one using the ReduceLROnPlateau scheduler
          def configure_optimizers(self):
              optimizer1 = Adam(...)
              optimizer2 = SGD(...)
              scheduler1 = ReduceLROnPlateau(optimizer1, ...)
              scheduler2 = LambdaLR(optimizer2, ...)
              return (
                  {
                      "optimizer": optimizer1,
                      "lr_scheduler": {
                          "scheduler": scheduler1,
                          "monitor": "metric_to_track",
                      },
                  },
                  {"optimizer": optimizer2, "lr_scheduler": scheduler2},
              )

      Metrics can be made available to monitor by simply logging it using
      ``self.log('metric_to_track', metric_val)`` in your :class:`~lightning.pytorch.core.module.LightningModule`.

      .. note::

         Some things to know:
         
         - Lightning calls ``.backward()`` and ``.step()`` automatically in case of automatic optimization.
         - If a learning rate scheduler is specified in ``configure_optimizers()`` with key
           ``"interval"`` (default "epoch") in the scheduler configuration, Lightning will call
           the scheduler's ``.step()`` method automatically in case of automatic optimization.
         - If you use 16-bit precision (``precision=16``), Lightning will automatically handle the optimizer.
         - If you use :class:`torch.optim.LBFGS`, Lightning handles the closure function automatically for you.
         - If you use multiple optimizers, you will have to switch to 'manual optimization' mode and step them
           yourself.
         - If you need to control how often the optimizer steps, override the :meth:`optimizer_step` hook.


   .. py:method:: load_from_checkpoint(checkpoint_path, map_location=None, hparams_file=None, strict=True, **kwargs)
      :classmethod:

      Primary way of loading a model from a checkpoint. When Lightning saves a checkpoint
      it stores the arguments passed to ``__init__``  in the checkpoint under ``"hyper_parameters"``.

      Any arguments specified through \*\*kwargs will override args stored in ``"hyper_parameters"``.

      :param checkpoint_path: Path to checkpoint. This can also be a URL, or file-like object
      :param map_location: If your checkpoint saved a GPU model and you now load on CPUs
                           or a different number of GPUs, use this to map to the new setup.
                           The behaviour is the same as in :func:`torch.load`.
      :param hparams_file: Optional path to a ``.yaml`` or ``.csv`` file with hierarchical structure
                           as in this example::

                               drop_prob: 0.2
                               dataloader:
                                   batch_size: 32

                           You most likely won't need this since Lightning will always save the hyperparameters
                           to the checkpoint.
                           However, if your checkpoint weights don't have the hyperparameters saved,
                           use this method to pass in a ``.yaml`` file with the hparams you'd like to use.
                           These will be converted into a :class:`~dict` and passed into your
                           :class:`LightningModule` for use.

                           If your model's ``hparams`` argument is :class:`~argparse.Namespace`
                           and ``.yaml`` file has hierarchical structure, you need to refactor your model to treat
                           ``hparams`` as :class:`~dict`.
      :param strict: Whether to strictly enforce that the keys in :attr:`checkpoint_path` match the keys
                     returned by this module's state dict.
      :param \**kwargs: Any extra keyword args needed to init the model. Can also be used to override saved
                        hyperparameter values.

      :returns: :class:`LightningModule` instance with loaded weights and hyperparameters (if available).

      .. note::

         ``load_from_checkpoint`` is a **class** method. You should use your :class:`LightningModule`
         **class** to call it instead of the :class:`LightningModule` instance.

      Example::

          # load weights without mapping ...
          model = MyLightningModule.load_from_checkpoint('path/to/checkpoint.ckpt')

          # or load weights mapping all weights from GPU 1 to GPU 0 ...
          map_location = {'cuda:1':'cuda:0'}
          model = MyLightningModule.load_from_checkpoint(
              'path/to/checkpoint.ckpt',
              map_location=map_location
          )

          # or load weights and hyperparameters from separate files.
          model = MyLightningModule.load_from_checkpoint(
              'path/to/checkpoint.ckpt',
              hparams_file='/path/to/hparams_file.yaml'
          )

          # override some of the params with new values
          model = MyLightningModule.load_from_checkpoint(
              PATH,
              num_layers=128,
              pretrained_ckpt_path=NEW_PATH,
          )

          # predict
          pretrained_model.eval()
          pretrained_model.freeze()
          y_hat = pretrained_model(x)



