:py:mod:`models.modules.readout`
================================

.. py:module:: models.modules.readout


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   models.modules.readout.ReadoutProto
   models.modules.readout.Readout
   models.modules.readout.ReadoutFFNBase
   models.modules.readout.RegressionFFN
   models.modules.readout.MveFFN
   models.modules.readout.EvidentialFFN
   models.modules.readout.BinaryClassificationFFNBase
   models.modules.readout.BinaryClassificationFFN
   models.modules.readout.BinaryDirichletFFN
   models.modules.readout.MulticlassClassificationFFN
   models.modules.readout.MulticlassDirichletFFN
   models.modules.readout.SpectralFFN




Attributes
~~~~~~~~~~

.. autoapisummary::

   models.modules.readout.ReadoutRegistry


.. py:data:: ReadoutRegistry

   

.. py:class:: ReadoutProto


   Bases: :py:obj:`Protocol`

   Base class for protocol classes.

   Protocol classes are defined as::

       class Proto(Protocol):
           def meth(self) -> int:
               ...

   Such classes are primarily used with static type checkers that recognize
   structural subtyping (static duck-typing), for example::

       class C:
           def meth(self) -> int:
               return 0

       def func(x: Proto) -> int:
           return x.meth()

       func(C())  # Passes static type check

   See PEP 544 for details. Protocol classes decorated with
   @typing.runtime_checkable act as simple-minded runtime protocols that check
   only the presence of given attributes, ignoring their type signatures.
   Protocol classes can be generic, they are defined as::

       class GenProto(Protocol[T]):
           def meth(self) -> T:
               ...

   .. py:attribute:: input_dim
      :type: int

      the input dimension

   .. py:attribute:: output_dim
      :type: int

      the output dimension

   .. py:attribute:: n_tasks
      :type: int

      the number of tasks `t` to predict for each input

   .. py:attribute:: n_targets
      :type: int

      the number of targets `s` to predict for each task `t`

   .. py:attribute:: criterion
      :type: chemprop.v2.models.loss.LossFunction

      the loss function to use for training

   .. py:method:: forward(Z: torch.Tensor) -> torch.Tensor


   .. py:method:: train_step(Z: torch.Tensor) -> torch.Tensor



.. py:class:: Readout(*args, **kwargs)


   Bases: :py:obj:`torch.nn.Module`, :py:obj:`ReadoutProto`, :py:obj:`chemprop.v2.models.hparams.HasHParams`

   Base class for all neural network modules.

   Your models should also subclass this class.

   Modules can also contain other Modules, allowing to nest them in
   a tree structure. You can assign the submodules as regular attributes::

       import torch.nn as nn
       import torch.nn.functional as F

       class Model(nn.Module):
           def __init__(self):
               super().__init__()
               self.conv1 = nn.Conv2d(1, 20, 5)
               self.conv2 = nn.Conv2d(20, 20, 5)

           def forward(self, x):
               x = F.relu(self.conv1(x))
               return F.relu(self.conv2(x))

   Submodules assigned in this way will be registered, and will have their
   parameters converted too when you call :meth:`to`, etc.

   .. note::
       As per the example above, an ``__init__()`` call to the parent class
       must be made before assignment on the child.

   :ivar training: Boolean represents whether this module is in training or
                   evaluation mode.
   :vartype training: bool


.. py:class:: ReadoutFFNBase(n_tasks: int = 1, input_dim: int = DEFAULT_HIDDEN_DIM, hidden_dim: int = 300, n_layers: int = 1, dropout: float = 0, activation: str = 'relu', criterion: chemprop.v2.models.loss.LossFunction | None = None)


   Bases: :py:obj:`Readout`, :py:obj:`lightning.pytorch.core.mixins.HyperparametersMixin`

   Base class for all neural network modules.

   Your models should also subclass this class.

   Modules can also contain other Modules, allowing to nest them in
   a tree structure. You can assign the submodules as regular attributes::

       import torch.nn as nn
       import torch.nn.functional as F

       class Model(nn.Module):
           def __init__(self):
               super().__init__()
               self.conv1 = nn.Conv2d(1, 20, 5)
               self.conv2 = nn.Conv2d(20, 20, 5)

           def forward(self, x):
               x = F.relu(self.conv1(x))
               return F.relu(self.conv2(x))

   Submodules assigned in this way will be registered, and will have their
   parameters converted too when you call :meth:`to`, etc.

   .. note::
       As per the example above, an ``__init__()`` call to the parent class
       must be made before assignment on the child.

   :ivar training: Boolean represents whether this module is in training or
                   evaluation mode.
   :vartype training: bool

   .. py:property:: input_dim
      :type: int


   .. py:property:: output_dim
      :type: int


   .. py:property:: n_tasks
      :type: int


   .. py:method:: forward(Z: torch.Tensor) -> torch.Tensor


   .. py:method:: train_step(Z: torch.Tensor) -> torch.Tensor



.. py:class:: RegressionFFN(*args, loc: float | torch.Tensor = 0, scale: float | torch.Tensor = 1, **kwargs)


   Bases: :py:obj:`ReadoutFFNBase`

   Base class for all neural network modules.

   Your models should also subclass this class.

   Modules can also contain other Modules, allowing to nest them in
   a tree structure. You can assign the submodules as regular attributes::

       import torch.nn as nn
       import torch.nn.functional as F

       class Model(nn.Module):
           def __init__(self):
               super().__init__()
               self.conv1 = nn.Conv2d(1, 20, 5)
               self.conv2 = nn.Conv2d(20, 20, 5)

           def forward(self, x):
               x = F.relu(self.conv1(x))
               return F.relu(self.conv2(x))

   Submodules assigned in this way will be registered, and will have their
   parameters converted too when you call :meth:`to`, etc.

   .. note::
       As per the example above, an ``__init__()`` call to the parent class
       must be made before assignment on the child.

   :ivar training: Boolean represents whether this module is in training or
                   evaluation mode.
   :vartype training: bool

   .. py:attribute:: n_targets
      :value: 1

      

   .. py:method:: forward(Z: torch.Tensor) -> torch.Tensor


   .. py:method:: train_step(Z: torch.Tensor) -> torch.Tensor



.. py:class:: MveFFN(*args, loc: float | torch.Tensor = 0, scale: float | torch.Tensor = 1, **kwargs)


   Bases: :py:obj:`RegressionFFN`

   Base class for all neural network modules.

   Your models should also subclass this class.

   Modules can also contain other Modules, allowing to nest them in
   a tree structure. You can assign the submodules as regular attributes::

       import torch.nn as nn
       import torch.nn.functional as F

       class Model(nn.Module):
           def __init__(self):
               super().__init__()
               self.conv1 = nn.Conv2d(1, 20, 5)
               self.conv2 = nn.Conv2d(20, 20, 5)

           def forward(self, x):
               x = F.relu(self.conv1(x))
               return F.relu(self.conv2(x))

   Submodules assigned in this way will be registered, and will have their
   parameters converted too when you call :meth:`to`, etc.

   .. note::
       As per the example above, an ``__init__()`` call to the parent class
       must be made before assignment on the child.

   :ivar training: Boolean represents whether this module is in training or
                   evaluation mode.
   :vartype training: bool

   .. py:attribute:: n_targets
      :value: 2

      

   .. py:method:: forward(Z: torch.Tensor) -> torch.Tensor


   .. py:method:: train_step(Z: torch.Tensor) -> torch.Tensor



.. py:class:: EvidentialFFN(*args, loc: float | torch.Tensor = 0, scale: float | torch.Tensor = 1, **kwargs)


   Bases: :py:obj:`RegressionFFN`

   Base class for all neural network modules.

   Your models should also subclass this class.

   Modules can also contain other Modules, allowing to nest them in
   a tree structure. You can assign the submodules as regular attributes::

       import torch.nn as nn
       import torch.nn.functional as F

       class Model(nn.Module):
           def __init__(self):
               super().__init__()
               self.conv1 = nn.Conv2d(1, 20, 5)
               self.conv2 = nn.Conv2d(20, 20, 5)

           def forward(self, x):
               x = F.relu(self.conv1(x))
               return F.relu(self.conv2(x))

   Submodules assigned in this way will be registered, and will have their
   parameters converted too when you call :meth:`to`, etc.

   .. note::
       As per the example above, an ``__init__()`` call to the parent class
       must be made before assignment on the child.

   :ivar training: Boolean represents whether this module is in training or
                   evaluation mode.
   :vartype training: bool

   .. py:attribute:: n_targets
      :value: 4

      

   .. py:method:: forward(Z: torch.Tensor) -> torch.Tensor


   .. py:method:: train_step(Z: torch.Tensor) -> torch.Tensor



.. py:class:: BinaryClassificationFFNBase(n_tasks: int = 1, input_dim: int = DEFAULT_HIDDEN_DIM, hidden_dim: int = 300, n_layers: int = 1, dropout: float = 0, activation: str = 'relu', criterion: chemprop.v2.models.loss.LossFunction | None = None)


   Bases: :py:obj:`ReadoutFFNBase`

   Base class for all neural network modules.

   Your models should also subclass this class.

   Modules can also contain other Modules, allowing to nest them in
   a tree structure. You can assign the submodules as regular attributes::

       import torch.nn as nn
       import torch.nn.functional as F

       class Model(nn.Module):
           def __init__(self):
               super().__init__()
               self.conv1 = nn.Conv2d(1, 20, 5)
               self.conv2 = nn.Conv2d(20, 20, 5)

           def forward(self, x):
               x = F.relu(self.conv1(x))
               return F.relu(self.conv2(x))

   Submodules assigned in this way will be registered, and will have their
   parameters converted too when you call :meth:`to`, etc.

   .. note::
       As per the example above, an ``__init__()`` call to the parent class
       must be made before assignment on the child.

   :ivar training: Boolean represents whether this module is in training or
                   evaluation mode.
   :vartype training: bool


.. py:class:: BinaryClassificationFFN(n_tasks: int = 1, input_dim: int = DEFAULT_HIDDEN_DIM, hidden_dim: int = 300, n_layers: int = 1, dropout: float = 0, activation: str = 'relu', criterion: chemprop.v2.models.loss.LossFunction | None = None)


   Bases: :py:obj:`BinaryClassificationFFNBase`

   Base class for all neural network modules.

   Your models should also subclass this class.

   Modules can also contain other Modules, allowing to nest them in
   a tree structure. You can assign the submodules as regular attributes::

       import torch.nn as nn
       import torch.nn.functional as F

       class Model(nn.Module):
           def __init__(self):
               super().__init__()
               self.conv1 = nn.Conv2d(1, 20, 5)
               self.conv2 = nn.Conv2d(20, 20, 5)

           def forward(self, x):
               x = F.relu(self.conv1(x))
               return F.relu(self.conv2(x))

   Submodules assigned in this way will be registered, and will have their
   parameters converted too when you call :meth:`to`, etc.

   .. note::
       As per the example above, an ``__init__()`` call to the parent class
       must be made before assignment on the child.

   :ivar training: Boolean represents whether this module is in training or
                   evaluation mode.
   :vartype training: bool

   .. py:attribute:: n_targets
      :value: 1

      

   .. py:method:: forward(Z: torch.Tensor) -> torch.Tensor


   .. py:method:: train_step(Z: torch.Tensor) -> torch.Tensor



.. py:class:: BinaryDirichletFFN(n_tasks: int = 1, input_dim: int = DEFAULT_HIDDEN_DIM, hidden_dim: int = 300, n_layers: int = 1, dropout: float = 0, activation: str = 'relu', criterion: chemprop.v2.models.loss.LossFunction | None = None)


   Bases: :py:obj:`BinaryClassificationFFNBase`

   Base class for all neural network modules.

   Your models should also subclass this class.

   Modules can also contain other Modules, allowing to nest them in
   a tree structure. You can assign the submodules as regular attributes::

       import torch.nn as nn
       import torch.nn.functional as F

       class Model(nn.Module):
           def __init__(self):
               super().__init__()
               self.conv1 = nn.Conv2d(1, 20, 5)
               self.conv2 = nn.Conv2d(20, 20, 5)

           def forward(self, x):
               x = F.relu(self.conv1(x))
               return F.relu(self.conv2(x))

   Submodules assigned in this way will be registered, and will have their
   parameters converted too when you call :meth:`to`, etc.

   .. note::
       As per the example above, an ``__init__()`` call to the parent class
       must be made before assignment on the child.

   :ivar training: Boolean represents whether this module is in training or
                   evaluation mode.
   :vartype training: bool

   .. py:attribute:: n_targets
      :value: 2

      

   .. py:method:: forward(Z: torch.Tensor) -> torch.Tensor


   .. py:method:: train_step(Z: torch.Tensor) -> torch.Tensor



.. py:class:: MulticlassClassificationFFN(n_classes: int, n_tasks: int = 1, *args, **kwargs)


   Bases: :py:obj:`ReadoutFFNBase`

   Base class for all neural network modules.

   Your models should also subclass this class.

   Modules can also contain other Modules, allowing to nest them in
   a tree structure. You can assign the submodules as regular attributes::

       import torch.nn as nn
       import torch.nn.functional as F

       class Model(nn.Module):
           def __init__(self):
               super().__init__()
               self.conv1 = nn.Conv2d(1, 20, 5)
               self.conv2 = nn.Conv2d(20, 20, 5)

           def forward(self, x):
               x = F.relu(self.conv1(x))
               return F.relu(self.conv2(x))

   Submodules assigned in this way will be registered, and will have their
   parameters converted too when you call :meth:`to`, etc.

   .. note::
       As per the example above, an ``__init__()`` call to the parent class
       must be made before assignment on the child.

   :ivar training: Boolean represents whether this module is in training or
                   evaluation mode.
   :vartype training: bool

   .. py:attribute:: n_targets
      :value: 1

      

   .. py:method:: forward(Z: torch.Tensor) -> torch.Tensor


   .. py:method:: train_step(Z: torch.Tensor) -> torch.Tensor



.. py:class:: MulticlassDirichletFFN(n_classes: int, n_tasks: int = 1, *args, **kwargs)


   Bases: :py:obj:`MulticlassClassificationFFN`

   Base class for all neural network modules.

   Your models should also subclass this class.

   Modules can also contain other Modules, allowing to nest them in
   a tree structure. You can assign the submodules as regular attributes::

       import torch.nn as nn
       import torch.nn.functional as F

       class Model(nn.Module):
           def __init__(self):
               super().__init__()
               self.conv1 = nn.Conv2d(1, 20, 5)
               self.conv2 = nn.Conv2d(20, 20, 5)

           def forward(self, x):
               x = F.relu(self.conv1(x))
               return F.relu(self.conv2(x))

   Submodules assigned in this way will be registered, and will have their
   parameters converted too when you call :meth:`to`, etc.

   .. note::
       As per the example above, an ``__init__()`` call to the parent class
       must be made before assignment on the child.

   :ivar training: Boolean represents whether this module is in training or
                   evaluation mode.
   :vartype training: bool

   .. py:method:: forward(Z: torch.Tensor) -> torch.Tensor


   .. py:method:: train_step(Z: torch.Tensor) -> torch.Tensor



.. py:class:: SpectralFFN(*args, spectral_activation: str | None = 'softplus', **kwargs)


   Bases: :py:obj:`ReadoutFFNBase`

   Base class for all neural network modules.

   Your models should also subclass this class.

   Modules can also contain other Modules, allowing to nest them in
   a tree structure. You can assign the submodules as regular attributes::

       import torch.nn as nn
       import torch.nn.functional as F

       class Model(nn.Module):
           def __init__(self):
               super().__init__()
               self.conv1 = nn.Conv2d(1, 20, 5)
               self.conv2 = nn.Conv2d(20, 20, 5)

           def forward(self, x):
               x = F.relu(self.conv1(x))
               return F.relu(self.conv2(x))

   Submodules assigned in this way will be registered, and will have their
   parameters converted too when you call :meth:`to`, etc.

   .. note::
       As per the example above, an ``__init__()`` call to the parent class
       must be made before assignment on the child.

   :ivar training: Boolean represents whether this module is in training or
                   evaluation mode.
   :vartype training: bool

   .. py:attribute:: n_targets
      :value: 1

      


