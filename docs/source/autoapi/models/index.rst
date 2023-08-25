:py:mod:`models`
================

.. py:module:: models


Subpackages
-----------
.. toctree::
   :titlesonly:
   :maxdepth: 3

   models/index.rst
   modules/index.rst


Submodules
----------
.. toctree::
   :titlesonly:
   :maxdepth: 1

   hparams/index.rst
   loss/index.rst
   metrics/index.rst
   model/index.rst
   multi/index.rst
   schedulers/index.rst
   utils/index.rst


Package Contents
----------------

Classes
~~~~~~~

.. autoapisummary::

   models.MessagePassingBlock
   models.MessagePassingBlockBase
   models.MulticomponentMessagePassing
   models.AtomMessageBlock
   models.BondMessageBlock
   models.MPNN
   models.LossFunction
   models.Metric




Attributes
~~~~~~~~~~

.. autoapisummary::

   models.MetricRegistry


.. py:class:: MessagePassingBlock(*args, **kwargs)


   Bases: :py:obj:`torch.nn.Module`, :py:obj:`MessagePassingProto`, :py:obj:`chemprop.v2.models.hparams.HasHParams`

   A :class:`MessagePassingBlock` is encodes a batch of molecular graphs using message passing
   to learn vertex-level hidden representations.


.. py:class:: MessagePassingBlockBase(d_v = DEFAULT_ATOM_FDIM, d_e = DEFAULT_BOND_FDIM, d_h = DEFAULT_HIDDEN_DIM, bias = False, depth = 3, dropout = 0, activation = Activation.RELU, undirected = False, d_vd = None)


   Bases: :py:obj:`chemprop.v2.models.modules.message_passing.base.MessagePassingBlock`, :py:obj:`lightning.pytorch.core.mixins.HyperparametersMixin`

   The base message-passing block for atom- and bond-based message-passing schemes

   NOTE: this class is an abstract base class and cannot be instantiated

   :param d_v: the feature dimension of the vertices
   :type d_v: int, default=DEFAULT_ATOM_FDIM
   :param d_e: the feature dimension of the edges
   :type d_e: int, default=DEFAULT_BOND_FDIM
   :param d_h: the hidden dimension during message passing
   :type d_h: int, default=DEFAULT_HIDDEN_DIM
   :param bias: if `True`, add a bias term to the learned weight matrices
   :type bias: bool, defuault=False
   :param depth: the number of message passing iterations
   :type depth: int, default=3
   :param undirected: if `True`, pass messages on undirected edges
   :type undirected: bool, default=False
   :param dropout: the dropout probability
   :type dropout: float, default=0
   :param activation: the activation function to use
   :type activation: str, default="relu"
   :param d_vd: the dimension of additional vertex descriptors that will be concatenated to the hidden features before readout
   :type d_vd: int | None, default=None

   .. seealso::

      * :class:`AtomMessageBlock`

      * :class:`BondMessageBlock`

   .. py:property:: output_dim
      :type: int


   .. py:method:: finalize(M_v, V, V_d)

      Finalize message passing by (1) concatenating the final hidden representations `H_v`
      and the original vertex ``V`` and (2) further concatenating additional vertex descriptors
      ``V_d``, if provided.

      This function implements the following operation:

      .. math::
          H_v &= \mathtt{dropout} \left( \tau(\mathbf{W}_o(V \mathbin\Vert M_v)) \right) \\
          H_v &= \mathtt{dropout} \left( \tau(\mathbf{W}_d(H_v \mathbin\Vert V_d)) \right),

      where :math:`\tau` is the activation function, :math:`\Vert` is the concatenation operator,
      :math:`\mathbf{W}_o` and :math:`\mathbf{W}_d` are learned weight matrices, :math:`M_v` is
      the message matrix, :math:`V` is the original vertex feature matrix, and :math:`V_d` is an
      optional vertex descriptor matrix.

      :param M_v: a tensor of shape ``V x d_h`` containing the messages sent from each atom
      :type M_v: Tensor
      :param V: a tensor of shape ``V x d_v`` containing the original vertex features
      :type V: Tensor
      :param V_d: an optional tensor of shape ``V x d_vd`` containing additional vertex descriptors
      :type V_d: Tensor | None

      :returns: a tensor of shape ``V x (d_h + d_v [+ d_vd])`` containing the final hidden
                representations
      :rtype: Tensor

      :raises InvalidShapeError: if ``V_d`` is not of shape ``b x d_vd``, where ``b`` is the batch size and ``d_vd`` is
          the vertex descriptor dimension


   .. py:method:: build(d_v = DEFAULT_ATOM_FDIM, d_e = DEFAULT_BOND_FDIM, d_h = DEFAULT_HIDDEN_DIM, d_vd = None, bias = False)
      :abstractmethod:

      construct the weight matrices used in the message passing update functions

      :param d_v: the vertex feature dimension
      :type d_v: int
      :param d_e: the edge feature dimension
      :type d_e: int
      :param d_h: the hidden dimension during message passing
      :type d_h: int, default=300
      :param d_vd: the dimension of additional vertex descriptors that will be concatenated to the hidden
                   features before readout, if any
      :type d_vd: int | None, default=None
      :param bias: whether to add a learned bias to the matrices
      :type bias: bool, default=False

      :returns: **W_i, W_h, W_o, W_d** -- the input, hidden, output, and descriptor weight matrices, respectively, used in the
                message passing update functions. The descriptor weight matrix is `None` if no vertex
                dimension is supplied
      :rtype: tuple[nn.Module, nn.Module, nn.Module, nn.Module | None]


   .. py:method:: forward(bmg, V_d = None)
      :abstractmethod:

      Encode a batch of molecular graphs.

      :param bmg: a batch of :class:`BatchMolGraph`s to encode
      :type bmg: BatchMolGraph
      :param V_d: an optional tensor of shape ``V x d_vd`` containing additional descriptors for each atom
                  in the batch. These will be concatenated to the learned atomic descriptors and
                  transformed before the readout phase.
                  **NOTE**: recall that ``V`` is equal to ``num_atoms + 1``, so ``V_d`` must be 0-padded
                  in the 0th row.
      :type V_d: Tensor | None, default=None

      :returns: a tensor of shape ``b x d_h`` or ``b x (d_h + d_vd)`` containing the encoding of each
                molecule in the batch, depending on whether additional atom descriptors were provided
      :rtype: Tensor



.. py:class:: MulticomponentMessagePassing(blocks, n_components, shared = False)


   Bases: :py:obj:`torch.nn.Module`

   A `MulticomponentMessagePassing` performs message-passing on each individual input in a
   multicomponent input then concatenates the representation of each input to construct a
   global representation

   :param blocks: the invidual message-passing blocks for each input
   :type blocks: Sequence[MolecularMessagePassingBlock]
   :param n_components: the number of components in each input
   :type n_components: int
   :param shared: whether one block will be shared among all components in an input. If not, a separate
                  block will be learned for each component.
   :type shared: bool, default=False

   .. py:property:: output_dim
      :type: int


   .. py:method:: __len__()


   .. py:method:: forward(bmgs, V_ds)

      Encode the multicomponent inputs

      :param bmgs:
      :type bmgs: Iterable[BatchMolGraph]
      :param V_ds:
      :type V_ds: Iterable[Tensor | None]

      :returns: a list of tensors of shape `b x d_i` containing the respective encodings of the `i`th component, where `b` is the number of components in the batch, and `d_i` is the output dimension of the `i`th encoder
      :rtype: list[Tensor]



.. py:class:: AtomMessageBlock(d_v = DEFAULT_ATOM_FDIM, d_e = DEFAULT_BOND_FDIM, d_h = DEFAULT_HIDDEN_DIM, bias = False, depth = 3, dropout = 0, activation = Activation.RELU, undirected = False, d_vd = None)


   Bases: :py:obj:`MessagePassingBlockBase`

   A :class:`AtomMessageBlock` encodes a batch of molecular graphs by passing messages along
   atoms.

   It implements the following operation:

   .. math::

       h_v^{(0)} &= \tau \left( \mathbf{W}_i(x_v) \right) \\
       m_v^{(t)} &= \sum_{u \in \mathcal{N}(v)} h_u^{(t-1)} \mathbin\Vert e_{uv} \\
       h_v^{(t)} &= \tau\left(h_v^{(0)} + \mathbf{W}_h m_v^{(t-1)}\right) \\
       m_v^{(T)} &= \sum_{w \in \mathcal{N}(v)} h_w^{(T-1)} \\
       h_v^{(T)} &= \tau \left (\mathbf{W}_o \left( x_v \mathbin\Vert m_{v}^{(T)} \right)  \right),

   where :math:`\tau` is the activation function; :math:`\mathbf{W}_i`, :math:`\mathbf{W}_h`, and
   :math:`\mathbf{W}_o` are learned weight matrices; :math:`e_{vw}` is the feature vector of the
   bond between atoms :math:`v` and :math:`w`; :math:`x_v` is the feature vector of atom :math:`v`;
   :math:`h_v^{(t)}` is the hidden representation of atom :math:`v` at iteration :math:`t`;
   :math:`m_v^{(t)}` is the message received by atom :math:`v` at iteration :math:`t`; and
   :math:`t \in \{1, \dots, T\}` is the number of message passing iterations.

   .. py:method:: build(d_v = DEFAULT_ATOM_FDIM, d_e = DEFAULT_BOND_FDIM, d_h = DEFAULT_HIDDEN_DIM, d_vd = None, bias = False)

      construct the weight matrices used in the message passing update functions

      :param d_v: the vertex feature dimension
      :type d_v: int
      :param d_e: the edge feature dimension
      :type d_e: int
      :param d_h: the hidden dimension during message passing
      :type d_h: int, default=300
      :param d_vd: the dimension of additional vertex descriptors that will be concatenated to the hidden
                   features before readout, if any
      :type d_vd: int | None, default=None
      :param bias: whether to add a learned bias to the matrices
      :type bias: bool, default=False

      :returns: **W_i, W_h, W_o, W_d** -- the input, hidden, output, and descriptor weight matrices, respectively, used in the
                message passing update functions. The descriptor weight matrix is `None` if no vertex
                dimension is supplied
      :rtype: tuple[nn.Module, nn.Module, nn.Module, nn.Module | None]


   .. py:method:: forward(bmg, V_d = None)

      Encode a batch of molecular graphs.

      :param bmg: a batch of :class:`BatchMolGraph`s to encode
      :type bmg: BatchMolGraph
      :param V_d: an optional tensor of shape ``V x d_vd`` containing additional descriptors for each atom
                  in the batch. These will be concatenated to the learned atomic descriptors and
                  transformed before the readout phase.
                  **NOTE**: recall that ``V`` is equal to ``num_atoms + 1``, so ``V_d`` must be 0-padded
                  in the 0th row.
      :type V_d: Tensor | None, default=None

      :returns: a tensor of shape ``b x d_h`` or ``b x (d_h + d_vd)`` containing the encoding of each
                molecule in the batch, depending on whether additional atom descriptors were provided
      :rtype: Tensor



.. py:class:: BondMessageBlock(d_v = DEFAULT_ATOM_FDIM, d_e = DEFAULT_BOND_FDIM, d_h = DEFAULT_HIDDEN_DIM, bias = False, depth = 3, dropout = 0, activation = Activation.RELU, undirected = False, d_vd = None)


   Bases: :py:obj:`MessagePassingBlockBase`

   A :class:`BondMessageBlock` encodes a batch of molecular graphs by passing messages along
   directed bonds.

   It implements the following operation:

   .. math::

       h_{vw}^{(0)} &= \tau \left( \mathbf{W}_i(e_{vw}) \right) \\
       m_{vw}^{(t)} &= \sum_{u \in \mathcal{N}(v)\setminus w} h_{uv}^{(t-1)} \\
       h_{vw}^{(t)} &= \tau \left(h_v^{(0)} + \mathbf{W}_h m_{vw}^{(t-1)} \right) \\
       m_v^{(T)} &= \sum_{w \in \mathcal{N}(v)} h_w^{(T-1)} \\
       h_v^{(T)} &= \tau \left (\mathbf{W}_o \left( x_v \mathbin\Vert m_{v}^{(T)} \right) \right),

   where :math:`\tau` is the activation function; :math:`\mathbf{W}_i`, :math:`\mathbf{W}_h`, and
   :math:`\mathbf{W}_o` are learned weight matrices; :math:`e_{vw}` is the feature vector of the
   bond between atoms :math:`v` and :math:`w`; :math:`x_v` is the feature vector of atom :math:`v`;
   :math:`h_{vw}^{(t)}` is the hidden representation of the bond :math:`v \rightarrow w` at
   iteration :math:`t`; :math:`m_{vw}^{(t)}` is the message received by the bond :math:`v
   \rightarrow w` at iteration :math:`t`; and :math:`t \in \{1, \dots, T-1\}` is the number of
   message passing iterations.

   .. py:method:: build(d_v = DEFAULT_ATOM_FDIM, d_e = DEFAULT_BOND_FDIM, d_h = DEFAULT_HIDDEN_DIM, d_vd = None, bias = False)

      construct the weight matrices used in the message passing update functions

      :param d_v: the vertex feature dimension
      :type d_v: int
      :param d_e: the edge feature dimension
      :type d_e: int
      :param d_h: the hidden dimension during message passing
      :type d_h: int, default=300
      :param d_vd: the dimension of additional vertex descriptors that will be concatenated to the hidden
                   features before readout, if any
      :type d_vd: int | None, default=None
      :param bias: whether to add a learned bias to the matrices
      :type bias: bool, default=False

      :returns: **W_i, W_h, W_o, W_d** -- the input, hidden, output, and descriptor weight matrices, respectively, used in the
                message passing update functions. The descriptor weight matrix is `None` if no vertex
                dimension is supplied
      :rtype: tuple[nn.Module, nn.Module, nn.Module, nn.Module | None]


   .. py:method:: forward(bmg, V_d = None)

      Encode a batch of molecular graphs.

      :param bmg: a batch of :class:`BatchMolGraph`s to encode
      :type bmg: BatchMolGraph
      :param V_d: an optional tensor of shape ``V x d_vd`` containing additional descriptors for each atom
                  in the batch. These will be concatenated to the learned atomic descriptors and
                  transformed before the readout phase.
                  **NOTE**: recall that ``V`` is equal to ``num_atoms + 1``, so ``V_d`` must be 0-padded
                  in the 0th row.
      :type V_d: Tensor | None, default=None

      :returns: a tensor of shape ``b x d_h`` or ``b x (d_h + d_vd)`` containing the encoding of each
                molecule in the batch, depending on whether additional atom descriptors were provided
      :rtype: Tensor



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



.. py:class:: LossFunction


   Bases: :py:obj:`abc.ABC`, :py:obj:`chemprop.v2.utils.ReprMixin`

   Helper class that provides a standard way to create an ABC using
   inheritance.

   .. py:method:: __call__(preds, targets, mask, w_s, w_t, lt_mask, gt_mask)

      Calculate the mean loss function value given predicted and target values

      :param preds: a tensor of shape `b x (t * s)` (regression), `b x t` (binary classification), or
                    `b x t x c` (multiclass classification) containing the predictions, where `b` is the
                    batch size, `t` is the number of tasks to predict, `s` is the number of
                    targets to predict for each task, and `c` is the number of classes.
      :type preds: Tensor
      :param targets: a float tensor of shape `b x t` containing the target values
      :type targets: Tensor
      :param mask: a boolean tensor of shape `b x t` indicating whether the given prediction should be
                   included in the loss calculation
      :type mask: Tensor
      :param w_s: a tensor of shape `b` or `b x 1` containing the per-sample weight
      :type w_s: Tensor
      :param w_t: a tensor of shape `t` or `1 x t` containing the per-task weight
      :type w_t: Tensor
      :param lt_mask:
      :type lt_mask: Tensor
      :param gt_mask:
      :type gt_mask: Tensor

      :returns: a scalar containing the fully reduced loss
      :rtype: Tensor


   .. py:method:: forward(preds, targets, mask, w_s, w_t, lt_mask, gt_mask)
      :abstractmethod:

      Calculate a tensor of shape `b x t` containing the unreduced loss values.



.. py:class:: Metric


   Bases: :py:obj:`chemprop.v2.models.loss.LossFunction`

   Helper class that provides a standard way to create an ABC using
   inheritance.

   .. py:attribute:: minimize
      :type: bool
      :value: True

      

   .. py:method:: __call__(preds, targets, mask, w_s, w_t, lt_mask, gt_mask)

      Calculate the mean loss function value given predicted and target values

      :param preds: a tensor of shape `b x (t * s)` (regression), `b x t` (binary classification), or
                    `b x t x c` (multiclass classification) containing the predictions, where `b` is the
                    batch size, `t` is the number of tasks to predict, `s` is the number of
                    targets to predict for each task, and `c` is the number of classes.
      :type preds: Tensor
      :param targets: a float tensor of shape `b x t` containing the target values
      :type targets: Tensor
      :param mask: a boolean tensor of shape `b x t` indicating whether the given prediction should be
                   included in the loss calculation
      :type mask: Tensor
      :param w_s: a tensor of shape `b` or `b x 1` containing the per-sample weight
      :type w_s: Tensor
      :param w_t: a tensor of shape `t` or `1 x t` containing the per-task weight
      :type w_t: Tensor
      :param lt_mask:
      :type lt_mask: Tensor
      :param gt_mask:
      :type gt_mask: Tensor

      :returns: a scalar containing the fully reduced loss
      :rtype: Tensor


   .. py:method:: forward(preds, targets, mask, lt_mask, gt_mask)
      :abstractmethod:

      Calculate a tensor of shape `b x t` containing the unreduced loss values.



.. py:data:: MetricRegistry

   

