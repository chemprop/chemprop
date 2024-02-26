:py:mod:`chemprop.chemprop.nn.agg`
==================================

.. py:module:: chemprop.chemprop.nn.agg


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   chemprop.chemprop.nn.agg.Aggregation
   chemprop.chemprop.nn.agg.MeanAggregation
   chemprop.chemprop.nn.agg.SumAggregation
   chemprop.chemprop.nn.agg.NormAggregation
   chemprop.chemprop.nn.agg.AttentiveAggregation




Attributes
~~~~~~~~~~

.. autoapisummary::

   chemprop.chemprop.nn.agg.AggregationRegistry


.. py:class:: Aggregation(dim = 0, *args, **kwargs)


   Bases: :py:obj:`torch.nn.Module`, :py:obj:`chemprop.nn.hparams.HasHParams`

   An :class:`Aggregation` aggregates the node-level representations of a batch of graphs into
   a batch of graph-level representations

   .. note::
       this class is abstract and cannot be instantiated.

   .. seealso:: :class:`~chemprop.v2.models.modules.agg.MeanAggregation`, :class:`~chemprop.v2.models.modules.agg.SumAggregation`, :class:`~chemprop.v2.models.modules.agg.NormAggregation`

   .. py:method:: forward(H, batch)
      :abstractmethod:

      Aggregate the graph-level representations of a batch of graphs into their respective
      global representations

      NOTE: it is possible for a graph to have 0 nodes. In this case, the representation will be
      a zero vector of length `d` in the final output.

      :param H: A tensor of shape ``V x d`` containing the batched node-level representations of ``n``
                graphs
      :type H: Tensor
      :param batch: a tensor of shape ``V`` containing the index of the graph a given vertex corresponds to
      :type batch: Tensor

      :returns: a tensor of shape ``n x d`` containing the graph-level representations
      :rtype: Tensor



.. py:data:: AggregationRegistry

   

.. py:class:: MeanAggregation(dim = 0, *args, **kwargs)


   Bases: :py:obj:`Aggregation`

   Average the graph-level representation:

   .. math::
       \mathbf h = \frac{1}{|V|} \sum_{v \in V} \mathbf h_v

   .. py:method:: forward(H, batch)

      Aggregate the graph-level representations of a batch of graphs into their respective
      global representations

      NOTE: it is possible for a graph to have 0 nodes. In this case, the representation will be
      a zero vector of length `d` in the final output.

      :param H: A tensor of shape ``V x d`` containing the batched node-level representations of ``n``
                graphs
      :type H: Tensor
      :param batch: a tensor of shape ``V`` containing the index of the graph a given vertex corresponds to
      :type batch: Tensor

      :returns: a tensor of shape ``n x d`` containing the graph-level representations
      :rtype: Tensor



.. py:class:: SumAggregation(dim = 0, *args, **kwargs)


   Bases: :py:obj:`Aggregation`

   Sum the graph-level representation:

   .. math::
       \mathbf h = \sum_{v \in V} \mathbf h_v


   .. py:method:: forward(H, batch)

      Aggregate the graph-level representations of a batch of graphs into their respective
      global representations

      NOTE: it is possible for a graph to have 0 nodes. In this case, the representation will be
      a zero vector of length `d` in the final output.

      :param H: A tensor of shape ``V x d`` containing the batched node-level representations of ``n``
                graphs
      :type H: Tensor
      :param batch: a tensor of shape ``V`` containing the index of the graph a given vertex corresponds to
      :type batch: Tensor

      :returns: a tensor of shape ``n x d`` containing the graph-level representations
      :rtype: Tensor



.. py:class:: NormAggregation(dim = 0, *args, norm = 100, **kwargs)


   Bases: :py:obj:`SumAggregation`

   Sum the graph-level representation and divide by a normalization constant:

   .. math::
       \mathbf h = \frac{1}{c} \sum_{v \in V} \mathbf h_v

   .. py:method:: forward(H, batch)

      Aggregate the graph-level representations of a batch of graphs into their respective
      global representations

      NOTE: it is possible for a graph to have 0 nodes. In this case, the representation will be
      a zero vector of length `d` in the final output.

      :param H: A tensor of shape ``V x d`` containing the batched node-level representations of ``n``
                graphs
      :type H: Tensor
      :param batch: a tensor of shape ``V`` containing the index of the graph a given vertex corresponds to
      :type batch: Tensor

      :returns: a tensor of shape ``n x d`` containing the graph-level representations
      :rtype: Tensor



.. py:class:: AttentiveAggregation(dim = 0, *args, output_size, **kwargs)


   Bases: :py:obj:`Aggregation`

   An :class:`Aggregation` aggregates the node-level representations of a batch of graphs into
   a batch of graph-level representations

   .. note::
       this class is abstract and cannot be instantiated.

   .. seealso:: :class:`~chemprop.v2.models.modules.agg.MeanAggregation`, :class:`~chemprop.v2.models.modules.agg.SumAggregation`, :class:`~chemprop.v2.models.modules.agg.NormAggregation`

   .. py:method:: forward(H, batch)

      Aggregate the graph-level representations of a batch of graphs into their respective
      global representations

      NOTE: it is possible for a graph to have 0 nodes. In this case, the representation will be
      a zero vector of length `d` in the final output.

      :param H: A tensor of shape ``V x d`` containing the batched node-level representations of ``n``
                graphs
      :type H: Tensor
      :param batch: a tensor of shape ``V`` containing the index of the graph a given vertex corresponds to
      :type batch: Tensor

      :returns: a tensor of shape ``n x d`` containing the graph-level representations
      :rtype: Tensor



