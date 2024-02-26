:py:mod:`chemprop.chemprop.models.multi`
========================================

.. py:module:: chemprop.chemprop.models.multi


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   chemprop.chemprop.models.multi.MulticomponentMPNN




.. py:class:: MulticomponentMPNN(message_passing, agg, predictor, batch_norm = True, metrics = None, w_t = None, warmup_epochs = 2, init_lr = 0.0001, max_lr = 0.001, final_lr = 0.0001)


   Bases: :py:obj:`chemprop.models.model.MPNN`

   An :class:`MPNN` is a sequence of message passing layers, an aggregation routine, and a
   predictor routine.

   The first two modules calculate learned fingerprints from an input molecule
   reaction graph, and the final module takes these leared fingerprints as input to calculate a
   final prediction. I.e., the following operation:

   .. math::
       \mathtt{MPNN}(\mathcal{G}) =
           \mathtt{predictor}(\mathtt{agg}(\mathtt{message\_passing}(\mathcal{G})))

   The full model is trained end-to-end.

   :param message_passing: the message passing block to use to calculate learned fingerprints
   :type message_passing: MessagePassing
   :param agg: the aggregation operation to use during molecule-level predictor
   :type agg: Aggregation
   :param predictor: the function to use to calculate the final prediction
   :type predictor: Predictor
   :param batch_norm: if `True`, apply batch normalization to the output of the aggregation operation
   :type batch_norm: bool, default=True
   :param metrics: the metrics to use to evaluate the model during training and evaluation
   :type metrics: Iterable[Metric] | None, default=None
   :param w_t: the weights to use for each task during training. If `None`, use uniform weights
   :type w_t: Tensor | None, default=None
   :param warmup_epochs: the number of epochs to use for the learning rate warmup
   :type warmup_epochs: int, default=2
   :param init_lr: the initial learning rate
   :type init_lr: int, default=1e-4
   :param max_lr: the maximum learning rate
   :type max_lr: float, default=1e-3
   :param final_lr: the final learning rate
   :type final_lr: float, default=1e-4

   :raises ValueError: if the output dimension of the message passing block does not match the input dimension of
       the predictor function

   .. py:method:: fingerprint(bmgs, V_ds, X_f = None)

      the learned fingerprints for the input molecules


   .. py:method:: load_from_checkpoint(checkpoint_path, map_location=None, hparams_file=None, strict=True, **kwargs)
      :classmethod:

      Primary way of loading a model from a checkpoint. When Lightning saves a checkpoint it stores the arguments
      passed to ``__init__``  in the checkpoint under ``"hyper_parameters"``.

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
         **class** to call it instead of the :class:`LightningModule` instance, or a
         ``TypeError`` will be raised.

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



   .. py:method:: load_from_file(model_path, map_location=None, strict=True)
      :classmethod:



