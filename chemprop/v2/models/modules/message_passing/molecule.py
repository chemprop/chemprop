from __future__ import annotations

from abc import abstractmethod
from typing import Optional

import torch
from torch import Tensor, nn

from chemprop.v2.exceptions import InvalidShapeError
from chemprop.v2.featurizers import BatchMolGraph, _DEFAULT_ATOM_FDIM, _DEFAULT_BOND_FDIM
from chemprop.v2.models.modules.readout import build_readout
from chemprop.v2.models.modules.message_passing.base import MessagePassingBlock
from chemprop.v2.models.utils import get_activation_function

MolecularInput = tuple[BatchMolGraph, Optional[Tensor]]


class MolecularMessagePassingBlock(MessagePassingBlock):
    """The message-passing block in an MPNN operating on molecules

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
    aggregation : Union[str, Aggregation], default=Aggregation.MEAN
        the aggregation function to use during readout
    d_vd : Optional[int], default=None
        the dimension of additional vertex descriptors that will be concatenated to the hidden features before readout

    See also
    --------
    `AtomMessageEncoder`

    `BondMessageEncoder`
    """

    def __init__(
        self,
        d_v: int,
        d_e: int,
        d_h: int = 300,
        bias: bool = False,
        depth: int = 3,
        undirected: bool = False,
        # layers_per_message: int = 1,
        dropout: float = 0,
        activation: str = "relu",
        aggregation: str = "mean",
        norm: float = 100,
        d_vd: Optional[int] = None,
    ):
        super().__init__()

        self.depth = depth
        self.undirected = undirected
        # self.layers_per_message = 1

        self.dropout = nn.Dropout(dropout)
        self.act = get_activation_function(activation)
        self.readout = build_readout(aggregation, norm)

        self.cached_zero_vector = nn.Parameter(torch.zeros(d_h), requires_grad=False)

        self.__output_dim = d_h

        if d_vd is not None:
            self.d_vd = d_vd
            self.__output_dim += d_vd
            self.fc_vd = nn.Linear(d_h + d_vd, d_h + d_vd)

        self.setup_weight_matrices(d_v, d_e, d_h, bias)

    @abstractmethod
    def setup_weight_matrices(self, d_v: int, d_e: int, d_h: int = 300, bias: bool = False):
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
        """

    @property
    def output_dim(self) -> int:
        return self.__output_dim

    def cat_descriptors(self, H_v: Tensor, X_vd: Tensor) -> Tensor:
        """Concatenate the atom descriptors `X_vd` onto the hidden representations `H_v`

        Parameters
        ----------
        H_v : Tensor
            a tensor of shape `V x d_h` containing the hidden representation of each atom
        X_vd : Tensor
            a tensor of shape `V x d_vd` containing additional descriptors for each atom

        Returns
        -------
        Tensor
            a tensor of shape `V x (d_h + d_vd)` containing the transformed hidden representations

        Raises
        ------
        InvalidShapeError
            if `X_vd` is of incorrect shape
        """
        try:
            H_vd = torch.cat((H_v, X_vd), 1)
            H_v = self.fc_vd(H_vd)
        except RuntimeError:
            raise InvalidShapeError("X_vd", X_vd.shape, [len(H_v), self.d_vd])

        H_v = self.dropout(H_v)

        return H_v  # V x (d_h + d_vd)

    @abstractmethod
    def forward(self, bmg: BatchMolGraph, X_vd: Optional[Tensor] = None) -> Tensor:
        """Encode a batch of molecular graphs.

        Parameters
        ----------
        bmg: BatchMolGraph
            the batch of `MolGraphs` to encode
        X_vd : Optional[Tensor], default=None
            an optional tensor of shape `V x d_vd` containing additional descriptors for each atom
            in the batch. These will be concatenated to the learned atomic descriptors and
            transformed before the readout phase. NOTE: recall that `V` is equal to `num_atoms` + 1,
            so if provided, this tensor must be 0-padded in the 0th row.

        Returns
        -------
        Tensor
            a tensor of shape `B x d_h` or `B x (d_h + d_vd)` containing the encoding of each
            molecule in the batch, depending on whether additional atom descriptors were provided
        """


class BondMessageBlock(MolecularMessagePassingBlock):
    def setup_weight_matrices(self, d_v: int, d_e: int, d_h: int = 300, bias: bool = False):
        self.W_i = nn.Linear(d_e, d_h, bias)
        self.W_h = nn.Linear(d_h, d_h, bias)
        self.W_o = nn.Linear(d_v + d_h, d_h)

    def forward(self, bmg: BatchMolGraph, X_vd: Optional[Tensor] = None) -> Tensor:
        H_0 = self.W_i(bmg.X_e)  # E x d_h

        H_e = self.act(H_0)
        for _ in range(1, self.depth):
            if self.undirected:
                H_e = (H_e + H_e[bmg.b2revb]) / 2

            M_e = H_e[bmg.a2b].sum(1)[bmg.b2a] - H_e[bmg.b2revb]  # E x d_h

            H_e = H_0 + self.W_h(M_e)  # E x d_h
            H_e = self.act(H_e)
            H_e = self.dropout(H_e)

        M_v = H_e[bmg.a2b].sum(1)  # V x d_h
        H_v = self.W_o(torch.cat((bmg.X_v, M_v), 1))  # V x d_h
        H_v = self.act(H_v)
        H_v = self.dropout(H_v)

        if X_vd is not None:
            H_v = self.cat_descriptors(H_v, X_vd)

        H = self.readout(H_v[1:], [n_a for _, n_a in bmg.a_scope])  # B x d_h + (d_vd)

        return H


class AtomMessageBlock(MolecularMessagePassingBlock):
    def setup_weight_matrices(self, d_v: int, d_e: int, d_h: int = 300, bias: bool = False):
        self.W_i = nn.Linear(d_v, d_h, bias)
        self.W_h = nn.Linear(d_e + d_h, d_h, bias)
        self.W_o = nn.Linear(d_v + d_h, d_h)

    def forward(self, bmg: BatchMolGraph, X_vd: Optional[Tensor] = None) -> Tensor:
        H_0 = self.W_i(bmg.X_v)  # V x d_h
        H_v = self.act(H_0)

        for _ in range(1, self.depth):
            if self.undirected:
                H_v = (H_v + H_v[bmg.b2revb]) / 2

            M_v_k = torch.cat((H_v[bmg.a2a], bmg.X_e[bmg.a2b]), 2)  # V x b x (d_h + d_e)
            M_v = M_v_k.sum(1)  # V x d_h + d_e

            H_v = H_0 + self.W_h(M_v)  # E x d_h
            H_v = self.act(H_v)
            H_v = self.dropout(H_v)

        M_v_k = H_v[bmg.a2a]
        M_v = M_v_k.sum(1)  # V x d_h

        H_v = self.act(self.W_o(torch.cat((bmg.X_v, M_v), 1)))  # V x d_h
        H_v = self.dropout(H_v)

        if X_vd is not None:
            H_v = self.cat_descriptors(H_v, X_vd)

        H = self.readout(H_v[1:], [n_a for _, n_a in bmg.a_scope])  # B x d_h (+ d_vd)

        return H


def molecule_block(
    d_v: int = _DEFAULT_ATOM_FDIM,
    d_e: int = _DEFAULT_BOND_FDIM,
    bond_messages: bool = True,
    *args,
    **kwargs,
) -> MolecularMessagePassingBlock:
    """Build a `MolecularMessagePassingBlock`

    NOTE: `d_v` and `d_e` should correspond to the `atom_fdim` and `bond_fdim` attributes of the
    `MoleculeFeaturizer` object that you will be using to prepare data. The default values should
    only be used if you are using a *default* `MoleculeFeaturizer` (i.e., `MoleculeFeaturizer()`)

    Parameters
    ----------
    d_v : int, default=_DEFAULT_ATOM_FDIM
        the dimension of the atom features
    d_e : int, default=_DEFAULT_BOND_FDIM
        the dimension of the bond features
    bond_messages : bool, optional
        whether to pass messages on bonds, default=True
    *args, **kwargs
        positional- and keyword-arguments to pass to the `MolecularMessagePassingBlock.__init__()`

    Returns
    -------
    MoleculeEncoder
    """
    encoder_cls = BondMessageBlock if bond_messages else AtomMessageBlock

    return encoder_cls(d_v, d_e, *args, **kwargs)
