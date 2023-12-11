from abc import abstractmethod

from lightning.pytorch.core.mixins import HyperparametersMixin
import torch
from torch import Tensor, nn

from chemprop.v2.conf import DEFAULT_ATOM_FDIM, DEFAULT_BOND_FDIM, DEFAULT_HIDDEN_DIM
from chemprop.v2.exceptions import InvalidShapeError
from chemprop.v2.data import BatchMolGraph
from chemprop.v2.nn.utils import Activation, get_activation_function
from chemprop.v2.nn.message_passing.base import MessagePassingBlock


class MessagePassingBlockBase(MessagePassingBlock, HyperparametersMixin):
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
    bias : bool, default=False
        if `True`, add a bias term to the learned weight matrices
    depth : int, default=3
        the number of message passing iterations
    undirected : bool, default=False
        if `True`, pass messages on undirected edges
    dropout : float, default=0
        the dropout probability
    activation : str, default="relu"
        the activation function to use
    d_vd : int | None, default=None
        the dimension of additional vertex descriptors that will be concatenated to the hidden features before readout

    See also
    --------
    * :class:`AtomMessageBlock`

    * :class:`BondMessageBlock`
    """

    def __init__(
        self,
        d_v: int = DEFAULT_ATOM_FDIM,
        d_e: int = DEFAULT_BOND_FDIM,
        d_h: int = DEFAULT_HIDDEN_DIM,
        bias: bool = False,
        depth: int = 3,
        dropout: float = 0,
        activation: str | Activation = Activation.RELU,
        undirected: bool = False,
        d_vd: int | None = None,
        # layers_per_message: int = 1,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.hparams["cls"] = self.__class__

        self.W_i, self.W_h, self.W_o, self.W_d = self.build(d_v, d_e, d_h, d_vd, bias)
        self.depth = depth
        self.undirected = undirected
        self.dropout = nn.Dropout(dropout)
        self.tau = get_activation_function(activation)

    @property
    def output_dim(self) -> int:
        return self.W_d.out_features if self.W_d is not None else self.W_o.out_features

    def finalize(self, M_v: Tensor, V: Tensor, V_d: Tensor | None) -> Tensor:
        r"""Finalize message passing by (1) concatenating the final hidden representations `H_v`
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

        Parameters
        ----------
        M_v : Tensor
            a tensor of shape ``V x d_h`` containing the messages sent from each atom
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
        H_v = self.W_o(torch.cat((V, M_v), 1))  # V x d_o
        H_v = self.tau(H_v)
        H_v = self.dropout(H_v)

        if V_d is not None:
            try:
                H_vd = torch.cat((H_v, V_d), 1)  # V x (d_o + d_vd)
                H_v = self.W_d(H_vd)  # V x (d_o + d_vd)
                H_v = self.dropout(H_v)
            except RuntimeError:
                raise InvalidShapeError("V_d", V_d.shape, [len(H_v), self.W_d.in_features])

        return H_v

    @abstractmethod
    def build(
        self,
        d_v: int = DEFAULT_ATOM_FDIM,
        d_e: int = DEFAULT_BOND_FDIM,
        d_h: int = DEFAULT_HIDDEN_DIM,
        d_vd: int | None = None,
        bias: bool = False,
    ) -> tuple[nn.Module, nn.Module, nn.Module, nn.Module | None]:
        """construct the weight matrices used in the message passing update functions

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
            **NOTE**: recall that ``V`` is equal to ``num_atoms + 1``, so ``V_d`` must be 0-padded
            in the 0th row.

        Returns
        -------
        Tensor
            a tensor of shape ``b x d_h`` or ``b x (d_h + d_vd)`` containing the encoding of each
            molecule in the batch, depending on whether additional atom descriptors were provided
        """


class BondMessageBlock(MessagePassingBlockBase):
    r"""A :class:`BondMessageBlock` encodes a batch of molecular graphs by passing messages along
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
    """

    def build(
        self,
        d_v: int = DEFAULT_ATOM_FDIM,
        d_e: int = DEFAULT_BOND_FDIM,
        d_h: int = DEFAULT_HIDDEN_DIM,
        d_vd: int | None = None,
        bias: bool = False,
    ):
        W_i = nn.Linear(d_e, d_h, bias)
        W_h = nn.Linear(d_h, d_h, bias)
        W_o = nn.Linear(d_v + d_h, d_h)
        W_d = nn.Linear(d_h + d_vd, d_h + d_vd) if d_vd is not None else None

        return W_i, W_h, W_o, W_d

    def forward(self, bmg: BatchMolGraph, V_d: Tensor | None = None) -> Tensor:
        H_0 = self.W_i(bmg.E)
        H_e = self.tau(H_0)

        for _ in range(1, self.depth):
            if self.undirected:
                H_e = (H_e + H_e[bmg.b2revb]) / 2

            # MESSAGE
            M_e_k = H_e[bmg.a2b]  # E x n_bonds x d_h
            M_e = M_e_k.sum(1)[bmg.b2a]  # E x d_h
            M_e = M_e - H_e[bmg.b2revb]  # subtract reverse bond message

            # UPDATE
            H_e = self.W_h(M_e)  # E x d_h
            H_e = self.tau(H_0 + H_e)
            H_e = self.dropout(H_e)

        M_v_k = H_e[bmg.a2b]
        M_v = M_v_k.sum(1)  # V x d_h

        return self.finalize(M_v, bmg.V, V_d)


class AtomMessageBlock(MessagePassingBlockBase):
    r"""A :class:`AtomMessageBlock` encodes a batch of molecular graphs by passing messages along
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

    def build(
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
        W_d = nn.Linear(d_h + d_vd, d_h + d_vd) if d_vd is not None else None

        return W_i, W_h, W_o, W_d

    def forward(self, bmg: BatchMolGraph, V_d: Tensor | None = None) -> Tensor:
        H_0 = self.W_i(bmg.V)  # V x d_h
        H_v = self.tau(H_0)

        for _ in range(1, self.depth):
            if self.undirected:
                H_v = (H_v + H_v[bmg.b2revb]) / 2

            # MESSAGE
            M_v_k = torch.cat((H_v[bmg.a2a], bmg.E[bmg.a2b]), 2)  # V x b x (d_h + d_e)
            M_v = M_v_k.sum(1)  # V x d_h + d_e

            # UPDATE
            H_v = self.W_h(M_v)  # E x d_h
            H_v = self.tau(H_0 + H_v)
            H_v = self.dropout(H_v)

        M_v_k = H_v[bmg.a2a]
        M_v = M_v_k.sum(1)  # V x d_h

        return self.finalize(M_v, bmg.V, V_d)
