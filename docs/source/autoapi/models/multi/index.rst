:py:mod:`models.multi`
======================

.. py:module:: models.multi


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   models.multi.MulticomponentMPNN




.. py:class:: MulticomponentMPNN(message_passing, agg, ffn, transform, loss_fn, metrics, task_weights = None, warmup_epochs = 2, num_lrs = 1, init_lr = 0.0001, max_lr = 0.001, final_lr = 0.0001)


   Bases: :py:obj:`chemprop.v2.models.model.MPNN`

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

   .. py:method:: fingerprint(bmgs, V_ds, X_f = None)

      the learned fingerprints for the input molecules


   .. py:method:: encoding(bmgs, V_ds, X_f = None)

      Calculate the encoding (i.e., final hidden representation) for the input molecules


   .. py:method:: forward(bmgs, V_ds, X_f = None)

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



