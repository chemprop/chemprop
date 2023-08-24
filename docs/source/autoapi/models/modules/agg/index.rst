:py:mod:`models.modules.agg`
============================

.. py:module:: models.modules.agg


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   models.modules.agg.Aggregation
   models.modules.agg.MeanAggregation
   models.modules.agg.SumAggregation
   models.modules.agg.NormAggregation




Attributes
~~~~~~~~~~

.. autoapisummary::

   models.modules.agg.AggregationRegistry


.. py:data:: AggregationRegistry

   

.. py:class:: Aggregation(dim: int = 0)


   Bases: :py:obj:`abc.ABC`, :py:obj:`torch.nn.Module`, :py:obj:`chemprop.v2.models.hparams.HasHParams`

   An :class:`Aggregation` aggregates the node-level representations of a batch of graphs into
   a batch of graph-level representations

   .. py:method:: forward(H: torch.Tensor, sizes: Sequence[int] | None) -> torch.Tensor

      Aggregate the graph-level representations of a batch of graphs into their respective
      global representations

      NOTE: it is possible for a graph to have 0 nodes. In this case, the representation will be
      a zero vector of length `d` in the final output.

      E.g., `H` is a tensor of shape ``10 x 4`` and ``sizes`` is equal to ``[3, 4, 3]``, then
      ``H[:3]``, ``H[3:7]``, and ``H[7:]`` correspond to the graph-level represenataions of the
      three individual graphs. The output of a call to ``forward()`` will be a tensor of shape
      ``3 x 4``

      :param H: A tensor of shape ``sum(sizes) x d`` containing the stacked node-level representations
                of ``len(sizes)`` graphs
      :type H: Tensor
      :param sizes: an list containing the number of nodes in each graph, respectively.
      :type sizes: Sequence[int]

      :returns: a tensor of shape ``len(sizes) x d`` containing the graph-level representation of each
                graph
      :rtype: Tensor

      :raises ValueError: if ``sum(sizes)`` is not equal to ``len(H_v)``


   .. py:method:: agg(H: torch.Tensor) -> torch.Tensor
      :abstractmethod:

      Aggregate the graph-level of a single graph into a vector

      :param H: A tensor of shape ``V x d`` containing the node-level representation of a graph with
                ``V`` nodes and node feature dimension ``d``
      :type H: Tensor

      :returns: a tensor of shape ``d`` containing the global representation of the input graph
      :rtype: Tensor



.. py:class:: MeanAggregation(dim: int = 0)


   Bases: :py:obj:`Aggregation`

   Average the graph-level representation

   .. py:method:: agg(H: torch.Tensor) -> torch.Tensor

      Aggregate the graph-level of a single graph into a vector

      :param H: A tensor of shape ``V x d`` containing the node-level representation of a graph with
                ``V`` nodes and node feature dimension ``d``
      :type H: Tensor

      :returns: a tensor of shape ``d`` containing the global representation of the input graph
      :rtype: Tensor



.. py:class:: SumAggregation(dim: int = 0)


   Bases: :py:obj:`Aggregation`

   Sum the graph-level representation

   .. py:method:: agg(H: torch.Tensor) -> torch.Tensor

      Aggregate the graph-level of a single graph into a vector

      :param H: A tensor of shape ``V x d`` containing the node-level representation of a graph with
                ``V`` nodes and node feature dimension ``d``
      :type H: Tensor

      :returns: a tensor of shape ``d`` containing the global representation of the input graph
      :rtype: Tensor



.. py:class:: NormAggregation(*args, norm: float = 100, **kwargs)


   Bases: :py:obj:`Aggregation`

   Sum the graph-level representation and divide by a normalization constant

   .. py:method:: agg(H: torch.Tensor) -> torch.Tensor

      Aggregate the graph-level of a single graph into a vector

      :param H: A tensor of shape ``V x d`` containing the node-level representation of a graph with
                ``V`` nodes and node feature dimension ``d``
      :type H: Tensor

      :returns: a tensor of shape ``d`` containing the global representation of the input graph
      :rtype: Tensor



