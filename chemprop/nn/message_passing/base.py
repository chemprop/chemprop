from abc import abstractmethod

from lightning.pytorch.core.mixins import HyperparametersMixin
import torch
from torch import Tensor, nn

from chemprop.conf import DEFAULT_ATOM_FDIM, DEFAULT_BOND_FDIM, DEFAULT_HIDDEN_DIM
from chemprop.data import BatchMolGraph
from chemprop.exceptions import InvalidShapeError
from chemprop.nn.message_passing.proto import MessagePassing
from chemprop.nn.transforms import GraphTransform, ScaleTransform
from chemprop.nn.utils import Activation, get_activation_function


class _MessagePassingBase(MessagePassing, HyperparametersMixin):
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
    undirected : bool, default=False
        if `True`, pass messages on undirected edges
    dropout : float, default=0.0
        the dropout probability
    activation : str, default="relu"
        the activation function to use
    d_vd : int | None, default=None
        the dimension of additional vertex descriptors that will be concatenated to the hidden features before readout

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
        V_d_transform: ScaleTransform | None = None,
        graph_transform: GraphTransform | None = None,
        # layers_per_message: int = 1,
    ):
        super().__init__()
        # manually add V_d_transform and graph_transform to hparams to suppress lightning's warning
        # about double saving their state_dict values.
        self.save_hyperparameters(ignore=["V_d_transform", "graph_transform"])
        self.hparams["V_d_transform"] = V_d_transform
        self.hparams["graph_transform"] = graph_transform
        self.hparams["cls"] = self.__class__

        self.W_i, self.W_h, self.W_o, self.W_d = self.setup(d_v, d_e, d_h, d_vd, bias)
        self.depth = depth
        self.undirected = undirected
        self.dropout = nn.Dropout(dropout)
        self.tau = get_activation_function(activation)
        self.V_d_transform = V_d_transform if V_d_transform is not None else nn.Identity()
        self.graph_transform = graph_transform if graph_transform is not None else nn.Identity()

    @property
    def output_dim(self) -> int:
        return self.W_d.out_features if self.W_d is not None else self.W_o.out_features

    @abstractmethod
    def setup(
        self,
        d_v: int = DEFAULT_ATOM_FDIM,
        d_e: int = DEFAULT_BOND_FDIM,
        d_h: int = DEFAULT_HIDDEN_DIM,
        d_vd: int | None = None,
        bias: bool = False,
    ) -> tuple[nn.Module, nn.Module, nn.Module, nn.Module | None]:
        """setup the weight matrices used in the message passing update functions

        Parameters
        ----------
        d_v : int
            the vertex feature dimension
        d_e : int
            the edge feature dimension
        d_h : int, default=300
            the hidden dimension during message passing
        d_vd : int | None, default=None
            the dimension of additional vertex descriptors that will be concatenated to the hidden
            features before readout, if any
        bias: bool, default=False
            whether to add a learned bias to the matrices

        Returns
        -------
        W_i, W_h, W_o, W_d : tuple[nn.Module, nn.Module, nn.Module, nn.Module | None]
            the input, hidden, output, and descriptor weight matrices, respectively, used in the
            message passing update functions. The descriptor weight matrix is `None` if no vertex
            dimension is supplied
        """

    @abstractmethod
    def initialize(self, bmg: BatchMolGraph) -> Tensor:
        """initialize the message passing scheme by calculating initial matrix of hidden features"""

    @abstractmethod
    def message(self, H_t: Tensor, bmg: BatchMolGraph):
        """Calculate the message matrix"""

    def update(self, M_t, H_0):
        """Calcualte the updated hidden for each edge"""
        H_t = self.W_h(M_t)
        H_t = self.tau(H_0 + H_t)
        H_t = self.dropout(H_t)

        return H_t

    def finalize(self, M: Tensor, V: Tensor, V_d: Tensor | None) -> Tensor:
        r"""Finalize message passing by (1) concatenating the final message ``M`` and the original
        vertex features ``V`` and (2) if provided, further concatenating additional vertex
        descriptors ``V_d``.

        This function implements the following operation:

        .. math::
            H &= \mathtt{dropout} \left( \tau(\mathbf{W}_o(V \mathbin\Vert M)) \right) \\
            H &= \mathtt{dropout} \left( \tau(\mathbf{W}_d(H \mathbin\Vert V_d)) \right),

        where :math:`\tau` is the activation function, :math:`\Vert` is the concatenation operator,
        :math:`\mathbf{W}_o` and :math:`\mathbf{W}_d` are learned weight matrices, :math:`M` is
        the message matrix, :math:`V` is the original vertex feature matrix, and :math:`V_d` is an
        optional vertex descriptor matrix.

        Parameters
        ----------
        M : Tensor
            a tensor of shape ``V x d_h`` containing the message vector of each vertex
        V : Tensor
            a tensor of shape ``V x d_v`` containing the original vertex features
        V_d : Tensor | None
            an optional tensor of shape ``V x d_vd`` containing additional vertex descriptors

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
                raise InvalidShapeError("V_d", V_d.shape, [len(H), self.W_d.in_features])

        return H

    def forward(self, bmg: BatchMolGraph, V_d: Tensor | None = None) -> Tensor:
        """Encode a batch of molecular graphs.

        Parameters
        ----------
        bmg: BatchMolGraph
            a batch of :class:`BatchMolGraph`s to encode
        V_d : Tensor | None, default=None
            an optional tensor of shape ``V x d_vd`` containing additional descriptors for each atom
            in the batch. These will be concatenated to the learned atomic descriptors and
            transformed before the readout phase.

        Returns
        -------
        Tensor
            a tensor of shape ``V x d_h`` or ``V x (d_h + d_vd)`` containing the encoding of each
            molecule in the batch, depending on whether additional atom descriptors were provided
        """
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
        return self.finalize(M, bmg.V, V_d)


class BondMessagePassing(_MessagePassingBase):
    r"""A :class:`BondMessagePassing` encodes a batch of molecular graphs by passing messages along
    directed bonds.

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
        bias: bool = False,
    ):
        W_i = nn.Linear(d_v + d_e, d_h, bias)
        W_h = nn.Linear(d_h, d_h, bias)
        W_o = nn.Linear(d_v + d_h, d_h)
        # initialize W_d only when d_vd is neither 0 nor None
        W_d = nn.Linear(d_h + d_vd, d_h + d_vd) if d_vd else None

        return W_i, W_h, W_o, W_d

    def initialize(self, bmg: BatchMolGraph) -> Tensor:
        return self.W_i(torch.cat([bmg.V[bmg.edge_index[0]], bmg.E], dim=1))

    def message(self, H: Tensor, bmg: BatchMolGraph) -> Tensor:
        index_torch = bmg.edge_index[1].unsqueeze(1).repeat(1, H.shape[1])
        M_all = torch.zeros(len(bmg.V), H.shape[1], dtype=H.dtype, device=H.device).scatter_reduce_(
            0, index_torch, H, reduce="sum", include_self=False
        )[bmg.edge_index[0]]
        M_rev = H[bmg.rev_edge_index]

        return M_all - M_rev


class AtomMessagePassing(_MessagePassingBase):
    r"""A :class:`AtomMessagePassing` encodes a batch of molecular graphs by passing messages along
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
    """

    def setup(
        self,
        d_v: int = DEFAULT_ATOM_FDIM,
        d_e: int = DEFAULT_BOND_FDIM,
        d_h: int = DEFAULT_HIDDEN_DIM,
        d_vd: int | None = None,
        bias: bool = False,
    ):
        W_i = nn.Linear(d_v, d_h, bias)
        W_h = nn.Linear(d_e + d_h, d_h, bias)
        W_o = nn.Linear(d_v + d_h, d_h)
        # initialize W_d only when d_vd is neither 0 nor None
        W_d = nn.Linear(d_h + d_vd, d_h + d_vd) if d_vd else None

        return W_i, W_h, W_o, W_d

    def initialize(self, bmg: BatchMolGraph) -> Tensor:
        return self.W_i(bmg.V[bmg.edge_index[0]])

    def message(self, H: Tensor, bmg: BatchMolGraph):
        H = torch.cat((H, bmg.E), dim=1)
        index_torch = bmg.edge_index[1].unsqueeze(1).repeat(1, H.shape[1])
        return torch.zeros(len(bmg.V), H.shape[1], dtype=H.dtype, device=H.device).scatter_reduce_(
            0, index_torch, H, reduce="sum", include_self=False
        )[bmg.edge_index[0]]
