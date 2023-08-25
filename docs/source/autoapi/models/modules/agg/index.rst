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

   

.. py:class:: Aggregation(dim = 0)


   Bases: :py:obj:`abc.ABC`, :py:obj:`torch.nn.Module`, :py:obj:`chemprop.v2.models.hparams.HasHParams`

   An :class:`Aggregation` aggregates the node-level representations of a batch of graphs into
   a batch of graph-level representations

   **NOTE**: this class is abstract and cannot be instantiated. Instead, you must use one of the
   concrete subclasses.

   .. seealso:: :class:`chemprop.v2.models.modules.agg.MeanAggregation`, :class:`chemprop.v2.models.modules.agg.SumAggregation`, :class:`chemprop.v2.models.modules.agg.NormAggregation`

   .. py:method:: forward(H, sizes)

      Aggregate the graph-level representations of a batch of graphs into their respective
      global representations

      NOTE: it is possible for a graph to have 0 nodes. In this case, the representation will be
      a zero vector of length `d` in the final output.

      :param H: A tensor of shape ``sum(sizes) x d`` containing the stacked node-level representations
                of ``len(sizes)`` graphs
      :type H: Tensor
      :param sizes: an list containing the number of nodes in each graph, respectively.
      :type sizes: Sequence[int]

      :returns: a tensor of shape ``len(sizes) x d`` containing the graph-level representation of each
                graph
      :rtype: Tensor

      :raises ValueError: if ``sum(sizes)`` is not equal to ``len(H_v)``

      .. rubric:: Examples

      **NOTE**: the following examples are for illustrative purposes only. In practice, you must
      use one of the concrete subclasses.

      1. A typical use-case:

      >>> H = torch.rand(10, 4)
      >>> sizes = [3, 4, 3]
      >>> agg = Aggregation()
      >>> agg(H, sizes).shape
      torch.Size([3, 4])

      2. A batch containing a graph with 0 nodes:

      >>> H = torch.rand(10, 4)
      >>> sizes = [3, 4, 0, 3]
      >>> agg = Aggregation()
      >>> agg(H, sizes).shape
      torch.Size([4, 4])


   .. py:method:: agg(H)
      :abstractmethod:

      Aggregate the graph-level of a single graph into a vector

      :param H: A tensor of shape ``V x d`` containing the node-level representation of a graph with
                ``V`` nodes and node feature dimension ``d``
      :type H: Tensor

      :returns: a tensor of shape ``d`` containing the global representation of the input graph
      :rtype: Tensor



.. py:class:: MeanAggregation(dim = 0)


   Bases: :py:obj:`Aggregation`

   Average the graph-level representation

   .. math::
       \mathbf h = \frac{1}{|V|} \sum_{v \in V} \mathbf h_v

   .. py:method:: agg(H)

      Aggregate the graph-level of a single graph into a vector

      :param H: A tensor of shape ``V x d`` containing the node-level representation of a graph with
                ``V`` nodes and node feature dimension ``d``
      :type H: Tensor

      :returns: a tensor of shape ``d`` containing the global representation of the input graph
      :rtype: Tensor



.. py:class:: SumAggregation(dim = 0)


   Bases: :py:obj:`Aggregation`

   Sum the graph-level representation

   .. math::
       \mathbf h = \sum_{v \in V} \mathbf h_v


   .. py:method:: agg(H)

      Aggregate the graph-level of a single graph into a vector

      :param H: A tensor of shape ``V x d`` containing the node-level representation of a graph with
                ``V`` nodes and node feature dimension ``d``
      :type H: Tensor

      :returns: a tensor of shape ``d`` containing the global representation of the input graph
      :rtype: Tensor



.. py:class:: NormAggregation(*args, norm = 100, **kwargs)


   Bases: :py:obj:`Aggregation`

   Sum the graph-level representation and divide by a normalization constant

   .. math::
       \mathbf h = \frac{1}{c} \sum_{v \in V} \mathbf h_v

   .. py:method:: agg(H)

      Aggregate the graph-level of a single graph into a vector

      :param H: A tensor of shape ``V x d`` containing the node-level representation of a graph with
                ``V`` nodes and node feature dimension ``d``
      :type H: Tensor

      :returns: a tensor of shape ``d`` containing the global representation of the input graph
      :rtype: Tensor



