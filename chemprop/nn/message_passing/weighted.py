import torch
from torch import Tensor, nn

from chemprop.conf import DEFAULT_ATOM_FDIM, DEFAULT_BOND_FDIM, DEFAULT_HIDDEN_DIM
from chemprop.data import BatchWeightedMolGraph
from chemprop.exceptions import InvalidShapeError
from chemprop.nn.message_passing.base import _MessagePassingBase
from chemprop.nn.message_passing.mixins import _WeightedBondMessagePassingMixin


class _WeightedMessagePassingBase(_MessagePassingBase):
    """The base message-passing block for weighted atom- and bond-based message-passing schemes

    NOTE: this class is an abstract base class and cannot be instantiated

    Parameters
    ----------
    d_v : int, default=DEFAULT_ATOM_FDIM
        the feature dimension of the vertices
    d_e : int, default=DEFAULT_BOND_FDIM
        the feature dimension of the edges
    d_h : int, default=DEFAULT_HIDDEN_DIM
        the hidden dimension during message passing
    bias : bool, defuault=False
        if `True`, add a bias term to the learned weight matrices
    depth : int, default=3
        the number of message passing iterations
    dropout : float, default=0.0
        the dropout probability
    activation : str | nn.Module, default="relu"
        the activation function to use
    undirected : bool, default=False
        if `True`, pass messages on undirected edges
    d_vd : int | None, default=None
        the dimension of additional vertex descriptors that will be concatenated to the hidden
        features before readout
    V_d_transform : ScaleTransform | None, default=None
        an optional transformation to apply to the additional vertex descriptors before concatenation
    graph_transform : GraphTransform | None, default=None
        an optional transformation to apply to the :class:`BatchMolGraph` before message passing. It
        is usually used to scale extra vertex and edge features.

    See also
    --------
    * :class:`WeightedBondMessagePassing`
    """

    def finalize(self, M: Tensor, V: Tensor, V_d: Tensor | None, V_w: Tensor | None) -> Tensor:
        r"""Finalize message passing by (1) concatenating the final message ``M`` and the original
        vertex features ``V`` and (2) if provided, further concatenating additional vertex
        descriptors ``V_d`` and weighting the output by the atom weights ``V_w``.

        This function implements the following operation:

        .. math::
            H &= \mathtt{dropout} \left( \tau(\mathbf{W}_o(V \mathbin\Vert M)) \right) \\
            H &= \mathtt{dropout} \left( \tau(\mathbf{W}_d(H \mathbin\Vert V_d)) \right),
            H &= w_v H,

        where :math:`\tau` is the activation function, :math:`\Vert` is the concatenation operator,
        :math:`\mathbf{W}_o` and :math:`\mathbf{W}_d` are learned weight matrices, :math:`M` is
        the message matrix, :math:`V` is the original vertex feature matrix, :math:`V_d` is an
        optional vertex descriptor matrix and :math: `w_v` is the atom weight matrix.

        Parameters
        ----------
        M : Tensor
            a tensor of shape ``V x d_h`` containing the message vector of each vertex
        V : Tensor
            a tensor of shape ``V x d_v`` containing the original vertex features
        V_d : Tensor | None
            an optional tensor of shape ``V x d_vd`` containing additional vertex descriptors
        V_w: Tensor
            a tensor of shape ``V`` containing the weights of each vertex

        Returns
        -------
        Tensor
            a tensor of shape ``V x (d_h + d_v [+ d_vd])`` containing the final hidden
            representations

        Raises
        ------
        InvalidShapeError
            if ``V_d`` is not of shape ``b x d_vd``, where ``b`` is the batch size and ``d_vd`` is
            the vertex descriptor dimension
        """
        H = self.W_o(torch.cat((V, M), dim=1))  # V x d_o
        H = self.tau(H)
        H = self.dropout(H)

        if V_d is not None:
            V_d = self.V_d_transform(V_d)
            try:
                H = self.W_d(torch.cat((H, V_d), dim=1))  # V x (d_o + d_vd)
                H = self.dropout(H)
            except RuntimeError:
                raise InvalidShapeError(
                    "V_d", V_d.shape, [len(H), self.W_d.in_features - self.W_o.out_features]
                )
        # Weight each atom feature vector by its atom weight
        H = torch.mul(V_w.unsqueeze(1), H)

        return H

    def forward(self, bmg: BatchWeightedMolGraph, V_d: Tensor | None = None) -> Tensor:
        bmg = self.graph_transform(bmg)
        H_0 = self.initialize(bmg)

        H = self.tau(H_0)
        for _ in range(1, self.depth):
            if self.undirected:
                H = (H + H[bmg.rev_edge_index]) / 2

            M = self.message(H, bmg)
            H = self.update(M, H_0)

        H = torch.mul(bmg.E_w.unsqueeze(1), H)
        index_torch = bmg.edge_index[1].unsqueeze(1).repeat(1, H.shape[1])
        M = torch.zeros(len(bmg.V), H.shape[1], dtype=H.dtype, device=H.device).scatter_reduce_(
            0, index_torch, H, reduce="sum", include_self=False
        )

        return self.finalize(M, bmg.V, V_d, bmg.V_w)


class WeightedBondMessagePassing(_WeightedBondMessagePassingMixin, _WeightedMessagePassingBase):
    r"""A :class:`WeightedBondMessagePassing` encodes a batch of weighted molecular graphs by passing weighted messages along directed bonds.

    It implements the following operation:

    .. math::

        h_{vw}^{(0)} &= \tau \left( \mathbf W_i(e_{vw}) \right) \\
        m_{vw}^{(t)} &= \sum_{u \in \mathcal N(v)\setminus w} w_{uv} h_{uv}^{(t-1)} \\
        h_{vw}^{(t)} &= \tau \left(h_v^{(0)} + \mathbf W_h m_{vw}^{(t-1)} \right) \\
        m_v^{(T)} &= \sum_{w \in \mathcal N(v)} w_{wv} h_w^{(T-1)} \\
        h_v^{(T)} &= \tau \left (\mathbf W_o \left( x_v \mathbin\Vert m_{v}^{(T)} \right) \right),

    where :math:`\tau` is the activation function; :math:`\mathbf W_i`, :math:`\mathbf W_h`, and
    :math:`\mathbf W_o` are learned weight matrices; :math:`e_{vw}` is the feature vector of the
    bond between atoms :math:`v` and :math:`w`; :math:`x_v` is the feature vector of atom :math:`v`;
    :math:`w_{uv}` is the bond weight of the bond :math:`u \rightarrow v`, according to the probability of :math:`v` being a neighbor of :math:`u`; :math:`h_{vw}^{(t)}` is the hidden representation of the bond :math:`v \rightarrow w` at iteration :math:`t`;
    :math:`m_{vw}^{(t)}` is the message received by the bond :math:`v \to w` at iteration :math:`t`; and :math:`t \in \{1, \dots, T-1\}` is the number of message passing iterations.
    """

    def setup(
        self,
        d_v: int = DEFAULT_ATOM_FDIM,
        d_e: int = DEFAULT_BOND_FDIM,
        d_h: int = DEFAULT_HIDDEN_DIM,
        d_vd: int | None = None,
        bias: bool = False,
    ):
        W_i = nn.Linear(d_v + d_e, d_h, bias)
        W_h = nn.Linear(d_h, d_h, bias)
        W_o = nn.Linear(d_v + d_h, d_h)
        W_d = nn.Linear(d_h + d_vd, d_h + d_vd) if d_vd else None

        return W_i, W_h, W_o, W_d
