from __future__ import annotations

from abc import abstractmethod

import torch
from torch import Tensor, nn

from chemprop.v2.conf import DEFAULT_ATOM_FDIM, DEFAULT_BOND_FDIM, DEFAULT_HIDDEN_DIM
from chemprop.v2.exceptions import InvalidShapeError
from chemprop.v2.featurizers import BatchMolGraph
from chemprop.v2.models.utils import get_activation_function
from chemprop.v2.models.modules.message_passing.base import MessagePassingBlock


class MessagePassingBlockBase(MessagePassingBlock):
    """The base message-passing block for atom- and bond-based MPNNs

    NOTE: this class is an abstract base class and cannot be instantiated

    Parameters
    ----------
    d_v : int
        the feature dimension of the vertices
    d_e : int
        the feature dimension of the edges
    d_h : int, default=30
        the hidden dimension during message passing
    bias : bool, optional
        whether to add a learned bias term to the weight matrices, by default False
    depth : int, default=3
        the number of message passing iterations
    undirected : bool, default=False
        whether messages should be bassed on undirected edges
    dropout : float, default=0
        the dropout probability
    activation : str, default="relu"
        the activation function to use
    aggregation : Aggregation | None, default=None
        the aggregation operation to use during molecule-level readout. If `None`, use `MeanAggregation`
    d_vd : int | None, default=None
        the dimension of additional vertex descriptors that will be concatenated to the hidden features before readout

    See also
    --------
    `AtomMessageBlock`

    `BondMessageBlock`
    """

    def __init__(
        self,
        d_v: int = DEFAULT_ATOM_FDIM,
        d_e: int = DEFAULT_BOND_FDIM,
        d_h: int = DEFAULT_HIDDEN_DIM,
        bias: bool = False,
        depth: int = 3,
        undirected: bool = False,
        dropout: float = 0,
        activation: str = "relu",
        d_vd: int | None = None,
        # layers_per_message: int = 1,
    ):
        super().__init__()

        self.depth = depth
        self.undirected = undirected
        # self.layers_per_message = 1

        self.dropout = nn.Dropout(dropout)
        self.tau = get_activation_function(activation)

        self.__output_dim = d_h

        if d_vd is not None:
            self.d_vd = d_vd
            self.__output_dim += d_vd
            self.fc_vd = nn.Linear(d_h + d_vd, d_h + d_vd)

        self.W_i, self.W_h, self.W_o = self.setup_weight_matrices(d_v, d_e, d_h, bias)

    @abstractmethod
    def setup_weight_matrices(
        self, d_v: int, d_e: int, d_h: int = 300, bias: bool = False
    ) -> tuple[nn.Module, nn.Module, nn.Module]:
        """set up the weight matrices used in the message passing udpate functions

        Parameters
        ----------
        d_v : int
            the vertex feature dimension
        d_e : int
            the edge feature dimension
        d_h : int, default=300
            the hidden dimension during message passing
        bias: bool, deafault=False
            whether to add a learned bias to the matrices

        Returns
        -------
        tuple[nn.Module, nn.Module, nn.Module]
            the input, hidden, and output weight matrices, respectively, used in the message
            passing update functions
        """

    @property
    def output_dim(self) -> int:
        return self.W_o.out_features

    def cat_descriptors(self, H_v: Tensor, V_d: Tensor) -> Tensor:
        """Concatenate the atom descriptors `V_d` onto the hidden representations `H_v`

        Parameters
        ----------
        H_v : Tensor
            a tensor of shape `V x d_h` containing the hidden representation of each atom
        V_d : Tensor
            a tensor of shape `V x d_vd` containing additional descriptors for each atom

        Returns
        -------
        Tensor
            a tensor of shape `V x (d_h + d_vd)` containing the transformed hidden representations

        Raises
        ------
        InvalidShapeError
            if `V_d` is not of shape `V x d_vd`
        """
        try:
            H_vd = torch.cat((H_v, V_d), 1)
            H_v = self.fc_vd(H_vd)
        except RuntimeError:
            raise InvalidShapeError("V_d", V_d.shape, [len(H_v), self.d_vd])

        return self.dropout(H_v)

    def finalize(self, M_v: Tensor, V: Tensor, V_d: Tensor | None) -> Tensor:
        H_v = self.W_o(torch.cat((V, M_v), 1))  # V x d_h
        H_v = self.tau(H_v)
        H_v = self.dropout(H_v)

        return H_v if V_d is None else self.cat_descriptors(H_v, V_d)

    @abstractmethod
    def forward(self, bmg: BatchMolGraph, V_d: Tensor | None = None) -> Tensor:
        """Encode a batch of molecular graphs.

        Parameters
        ----------
        bmg: BatchMolGraph
            the batch of `b` `MolGraphs` to encode
        V_d : Tensor | None, default=None
            an optional tensor of shape `V x d_vd` containing additional descriptors for each atom
            in the batch. These will be concatenated to the learned atomic descriptors and
            transformed before the readout phase. NOTE: recall that `V` is equal to `num_atoms + 1`,
            so if provided, this tensor must be 0-padded in the 0th row.

        Returns
        -------
        Tensor
            a tensor of shape `b x d_h` or `b x (d_h + d_vd)` containing the encoding of each
            molecule in the batch, depending on whether additional atom descriptors were provided
        """


class BondMessageBlock(MessagePassingBlockBase):
    def setup_weight_matrices(self, d_v: int, d_e: int, d_h: int = 300, bias: bool = False):
        W_i = nn.Linear(d_e, d_h, bias)
        W_h = nn.Linear(d_h, d_h, bias)
        W_o = nn.Linear(d_v + d_h, d_h)

        return W_i, W_h, W_o

    def forward(self, bmg: BatchMolGraph, V_d: Tensor | None = None) -> Tensor:
        H_0 = self.W_i(bmg.E)
        H_e = self.tau(H_0)

        for _ in range(1, self.depth):
            if self.undirected:
                H_e = (H_e + H_e[bmg.b2revb]) / 2

            # MESSAGE
            M_e_k = H_e[bmg.a2b]    # E x n_bonds x d_h
            M_e = M_e_k.sum(1)[bmg.b2a] # E x d_h
            M_e = M_e - H_e[bmg.b2revb]  # subtract reverse bond message

            # UPDATE
            H_e = self.W_h(M_e)  # E x d_h
            H_e = self.tau(H_0 + H_e)
            H_e = self.dropout(H_e)

        M_v_k = H_e[bmg.a2b]
        M_v = M_v_k.sum(1)  # V x d_h

        return self.finalize(M_v, bmg.V, V_d)


class AtomMessageBlock(MessagePassingBlockBase):
    def setup_weight_matrices(self, d_v: int, d_e: int, d_h: int = 300, bias: bool = False):
        W_i = nn.Linear(d_v, d_h, bias)
        W_h = nn.Linear(d_e + d_h, d_h, bias)
        W_o = nn.Linear(d_v + d_h, d_h)

        return W_i, W_h, W_o

    def forward(self, bmg: BatchMolGraph, V_d: Tensor | None = None) -> Tensor:
        H_0 = self.W_i(bmg.V)  # V x d_h
        H_v = self.tau(H_0)

        for _ in range(1, self.depth):
            if self.undirected:
                H_v = (H_v + H_v[bmg.b2revb]) / 2

            # aggregate messages
            M_v_k = torch.cat((H_v[bmg.a2a], bmg.E[bmg.a2b]), 2)  # V x b x (d_h + d_e)
            M_v = M_v_k.sum(1)  # V x d_h + d_e

            # UPDATE
            H_v = self.W_h(M_v)  # E x d_h
            H_v = self.tau(H_0 + H_v)
            H_v = self.dropout(H_v)

        M_v_k = H_v[bmg.a2a]
        M_v = M_v_k.sum(1)  # V x d_h

        return self.finalize(M_v, bmg.V, V_d)
