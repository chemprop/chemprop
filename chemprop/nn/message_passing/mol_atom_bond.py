from abc import abstractmethod

from lightning.pytorch.core.mixins import HyperparametersMixin
import torch
from torch import Tensor, nn

from chemprop.conf import DEFAULT_ATOM_FDIM, DEFAULT_BOND_FDIM, DEFAULT_HIDDEN_DIM
from chemprop.data import BatchMolGraph
from chemprop.exceptions import InvalidShapeError
from chemprop.nn.message_passing.mixins import _AtomMessagePassingMixin, _BondMessagePassingMixin
from chemprop.nn.message_passing.proto import MABMessagePassing
from chemprop.nn.transforms import GraphTransform, ScaleTransform
from chemprop.nn.utils import Activation, get_activation_function


class _MABMessagePassingBase(MABMessagePassing, HyperparametersMixin):
    """The base message-passing block for atom- and bond-based message-passing schemes

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
    activation : str, default="relu"
        the activation function to use
    undirected : bool, default=False
        if `True`, pass messages on undirected edges
    d_vd : int | None, default=None
        the dimension of additional vertex descriptors that will be concatenated to the hidden
        features before readout
    d_ed : int | None, default=None
        the dimension of additional edge descriptors that will be concatenated to the hidden
        features before readout
    V_d_transform : ScaleTransform | None, default=None
        an optional transformation to apply to the additional vertex descriptors before concatenation
    E_d_transform : ScaleTransform | None, default=None
        an optional transformation to apply to the additional edge descriptors before concatenation
    graph_transform : GraphTransform | None, default=None
        an optional transformation to apply to the :class:`BatchMolGraph` before message passing. It
        is usually used to scale extra vertex and edge features.
    return_vertex_embeddings : bool, default=True
        whether to return the learned vertex embeddings. If `False`, None is returned.
    return_edge_embeddings : bool, default=True
        whether to return the learned edge embeddings. If `False`, None is returned.

    See also
    --------
    * :class:`AtomMessagePassing`

    * :class:`BondMessagePassing`
    """

    def __init__(
        self,
        d_v: int = DEFAULT_ATOM_FDIM,
        d_e: int = DEFAULT_BOND_FDIM,
        d_h: int = DEFAULT_HIDDEN_DIM,
        bias: bool = False,
        depth: int = 3,
        dropout: float = 0.0,
        activation: str | Activation = Activation.RELU,
        undirected: bool = False,
        d_vd: int | None = None,
        d_ed: int | None = None,
        V_d_transform: ScaleTransform | None = None,
        E_d_transform: ScaleTransform | None = None,
        graph_transform: GraphTransform | None = None,
        return_vertex_embeddings: bool = True,
        return_edge_embeddings: bool = True,
    ):
        super().__init__()
        # manually add transforms to hparams to suppress lightning's warning
        # about double saving their state_dict values.
        self.save_hyperparameters(ignore=["V_d_transform", "E_d_transform", "graph_transform"])
        self.hparams["V_d_transform"] = V_d_transform
        self.hparams["E_d_transform"] = E_d_transform
        self.hparams["graph_transform"] = graph_transform
        self.hparams["cls"] = self.__class__

        self.return_vertex_embeddings = return_vertex_embeddings
        self.return_edge_embeddings = return_edge_embeddings

        self.W_i, self.W_h, self.W_vo, self.W_vd, self.W_eo, self.W_ed = self.setup(
            d_v, d_e, d_h, d_vd, d_ed, bias
        )
        self.depth = depth
        self.undirected = undirected
        self.dropout = nn.Dropout(dropout)
        self.tau = get_activation_function(activation)
        self.V_d_transform = V_d_transform if V_d_transform is not None else nn.Identity()
        self.E_d_transform = E_d_transform if E_d_transform is not None else nn.Identity()
        self.graph_transform = graph_transform if graph_transform is not None else nn.Identity()

    @property
    def output_dims(self) -> tuple[int | None, int | None]:
        """Returns the output dimensions of the vertex and edge embeddings."""
        return (
            self.W_d.out_features if self.W_d is not None else self.W_o.out_features,
            self.W_ed.out_features if self.W_ed is not None else self.W_o_b.out_features,
        )


class MixedBondMessagePassing(_MixedMessagePassingBase, _BondMessagePassingMixin):
    def setup(
        self,
        d_v: int = DEFAULT_ATOM_FDIM,
        d_e: int = DEFAULT_BOND_FDIM,
        d_h: int = DEFAULT_HIDDEN_DIM,
        d_vd: int | None = None,
        d_ed: int | None = None,
        bias: bool = False,
    ):
        W_i = nn.Linear(d_v + d_e, d_h, bias)
        W_h = nn.Linear(d_h, d_h, bias)
        W_o = nn.Linear(d_v + d_h, d_h)
        W_o_b = nn.Linear(d_e + d_h, d_h)
        W_d = nn.Linear(d_h + d_vd, d_h + d_vd) if d_vd else None
        W_ed = nn.Linear(d_h + d_ed, d_h + d_ed) if d_ed else None

        return W_i, W_h, W_o, W_d, W_o_b, W_ed

    def finalize(
        self, H: Tensor, M: Tensor, V: Tensor, E: Tensor, V_d: Tensor | None, E_d: Tensor | None
    ) -> tuple[Tensor]:
        H_v = self.W_o(torch.cat((V, M), dim=1))
        H_v = self.tau(H_v)
        H_v = self.dropout(H_v)
        H_b = self.W_o_b(torch.cat((E, H), dim=1))
        H_b = self.tau(H_b)
        H_b = self.dropout(H_b)

        if V_d is not None:
            V_d = self.V_d_transform(V_d)
            try:
                H_v = self.W_d(torch.cat((H_v, V_d), dim=1))  # V x (d_o + d_vd)
                H_v = self.dropout(H_v)
            except RuntimeError:
                raise InvalidShapeError("V_d", V_d.shape, [len(H_v), self.W_d.in_features])

        if E_d is not None:
            E_d = self.E_d_transform(E_d)
            try:
                H_b = self.W_ed(torch.cat((H_b, E_d), dim=1))
                H_b = self.dropout(H_b)
            except RuntimeError:
                raise InvalidShapeError("E_d", E_d.shape, [len(H_b), self.W_ed.in_features])

        return H_v, H_b

    def forward(
        self, bmg: BatchMolGraph, V_d: Tensor | None = None, E_d: Tensor | None = None
    ) -> tuple[Tensor | None, Tensor | None]:
        bmg = self.graph_transform(bmg)
        H_0 = self.initialize(bmg)

        H = self.tau(H_0)
        for _ in range(1, self.depth):
            if self.undirected:
                H = (H + H[bmg.rev_edge_index]) / 2

            M = self.message(H, bmg)
            H = self.update(M, H_0)

        index_torch = bmg.edge_index[1].unsqueeze(1).repeat(1, H.shape[1])
        M = torch.zeros(len(bmg.V), H.shape[1], dtype=H.dtype, device=H.device).scatter_reduce_(
            0, index_torch, H, reduce="sum", include_self=False
        )

        H_v = self.vertex_finalize(M, bmg.V, V_d) if self.return_vertex_embeddings else None
        H_e = self.edge_finalize(H, bmg.E, E_d) if self.return_edge_embeddings else None
        return H_v, H_e


class MABBondMessagePassing(_BondMessagePassingMixin, _MABMessagePassingBase):
    r"""A :class:`MABBondMessagePassing` encodes a batch of molecular graphs by passing messages
    along directed bonds.

    It implements the following operation:

    .. math::

        h_{vw}^{(0)} &= \tau \left( \mathbf W_i(e_{vw}) \right) \\
        m_{vw}^{(t)} &= \sum_{u \in \mathcal N(v)\setminus w} h_{uv}^{(t-1)} \\
        h_{vw}^{(t)} &= \tau \left(h_v^{(0)} + \mathbf W_h m_{vw}^{(t-1)} \right) \\
        m_v^{(T)} &= \sum_{w \in \mathcal N(v)} h_w^{(T-1)} \\
        h_v^{(T)} &= \tau \left (\mathbf W_o \left( x_v \mathbin\Vert m_{v}^{(T)} \right) \right),

    where :math:`\tau` is the activation function; :math:`\mathbf W_i`, :math:`\mathbf W_h`, and
    :math:`\mathbf W_o` are learned weight matrices; :math:`e_{vw}` is the feature vector of the
    bond between atoms :math:`v` and :math:`w`; :math:`x_v` is the feature vector of atom :math:`v`;
    :math:`h_{vw}^{(t)}` is the hidden representation of the bond :math:`v \rightarrow w` at
    iteration :math:`t`; :math:`m_{vw}^{(t)}` is the message received by the bond :math:`v
    \to w` at iteration :math:`t`; and :math:`t \in \{1, \dots, T-1\}` is the number of
    message passing iterations.
    """

    def setup(
        self,
        d_v: int = DEFAULT_ATOM_FDIM,
        d_e: int = DEFAULT_BOND_FDIM,
        d_h: int = DEFAULT_HIDDEN_DIM,
        d_vd: int | None = None,
        d_ed: int | None = None,
        bias: bool = False,
    ):
        W_i = nn.Linear(d_v + d_e, d_h, bias)
        W_h = nn.Linear(d_h, d_h, bias)
        W_vo = nn.Linear(d_v + d_h, d_h) if self.return_vertex_embeddings else None
        W_eo = nn.Linear(d_e + d_h, d_h) if self.return_edge_embeddings else None
        W_vd = nn.Linear(d_h + d_vd, d_h + d_vd) if d_vd else None
        W_ed = nn.Linear(d_h + d_ed, d_h + d_ed) if d_ed else None

        return W_i, W_h, W_vo, W_vd, W_eo, W_ed


class MABAtomMessagePassing(_AtomMessagePassingMixin, _MABMessagePassingBase):
    r"""A :class:`MABAtomMessagePassing` encodes a batch of molecular graphs by passing messages
    along atoms.

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
    """

    def setup(
        self,
        d_v: int = DEFAULT_ATOM_FDIM,
        d_e: int = DEFAULT_BOND_FDIM,
        d_h: int = DEFAULT_HIDDEN_DIM,
        d_vd: int | None = None,
        d_ed: int | None = None,
        bias: bool = False,
    ):
        W_i = nn.Linear(d_v, d_h, bias)
        W_h = nn.Linear(d_e + d_h, d_h, bias)
        W_o = nn.Linear(d_v + d_h, d_h)
        W_o_b = nn.Linear(d_e + d_h, d_h)
        W_d = nn.Linear(d_h + d_vd, d_h + d_vd) if d_vd else None
        W_ed = nn.Linear(d_h + d_ed, d_h + d_ed) if d_ed else None

        return W_i, W_h, W_o, W_d, W_o_b, W_ed

    def finalize(
        self, H: Tensor, M: Tensor, V: Tensor, E: Tensor, V_d: Tensor | None, E_d: Tensor | None
    ) -> tuple[Tensor]:
        H_v = self.W_o(torch.cat((V, M), dim=1))
        H_v = self.tau(H_v)
        H_v = self.dropout(H_v)
        H_b = self.W_o_b(torch.cat((E, H), dim=1))
        H_b = self.tau(H_b)
        H_b = self.dropout(H_b)

        if V_d is not None:
            V_d = self.V_d_transform(V_d)
            try:
                H_v = self.W_d(torch.cat((H_v, V_d), dim=1))  # V x (d_o + d_vd)
                H_v = self.dropout(H_v)
            except RuntimeError:
                raise InvalidShapeError("V_d", V_d.shape, [len(H_v), self.W_d.in_features])

        if E_d is not None:
            E_d = self.E_d_transform(E_d)
            try:
                H_b = self.W_ed(torch.cat((H_b, V_d), dim=1))
                H_b = self.dropout(H_b)
            except RuntimeError:
                raise InvalidShapeError("E_d", E_d.shape, [len(H_b), self.W_ed.in_features])

        return H_v, H_b

    def forward(
        self, bmg: BatchMolGraph, V_d: Tensor | None = None, E_d: Tensor | None = None
    ) -> tuple[Tensor]:
        bmg = self.graph_transform(bmg)
        H_0 = self.initialize(bmg)

        H = self.tau(H_0)
        for _ in range(1, self.depth):
            if self.undirected:
                H = (H + H[bmg.rev_edge_index]) / 2

            M = self.message(H, bmg)
            H = self.update(M, H_0)

        index_torch = bmg.edge_index[1].unsqueeze(1).repeat(1, H.shape[1])
        M = torch.zeros(len(bmg.V), H.shape[1], dtype=H.dtype, device=H.device).scatter_reduce_(
            0, index_torch, H, reduce="sum", include_self=False
        )
        return self.finalize(H, M, bmg.V, bmg.E, V_d, E_d)

    def initialize(self, bmg: BatchMolGraph) -> Tensor:
        return _AtomMessagePassingMixin.initialize(self, bmg)

    def message(self, H: Tensor, bmg: BatchMolGraph):
        return _AtomMessagePassingMixin.message(self, H, bmg)
