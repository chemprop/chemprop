from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable, Optional, Tuple

import torch
from torch import Tensor, nn

from chemprop.nn_utils import get_activation_function


class MPNEncoder(nn.Module, ABC):
    def __len__(self) -> int:
        """an alias for the output dimension of the encoder"""
        return self.output_dim

    @property
    @abstractmethod
    def output_dim(self) -> int:
        pass


class MoleculeEncoder(MPNEncoder, nn.Module, ABC):
    def __init__(
        self,
        bond_messages: bool,
        d_v: int,
        d_e: int,
        d_h: int = 300,
        bias: bool = False,
        depth: int = 3,
        undirected: bool = False,
        # layers_per_message: int,
        dropout: float = 0,
        activation: str = "relu",
        aggregation: str = "mean",
        atom_descriptors: Optional[str] = None,
        d_vd: Optional[int] = None,
    ):
        super().__init__()

        self.depth = depth
        self.undirected = undirected
        # self.layers_per_message = 1

        self.dropout = nn.Dropout(dropout)
        self.act = get_activation_function(activation)
        self.aggregation = aggregation

        self.cached_zero_vector = nn.Parameter(torch.zeros(d_h), requires_grad=False)

        self.__output_dim = d_h
        if bond_messages:
            self.W_i = nn.Linear(d_e, d_h, bias)
            self.W_h = nn.Linear(d_h, d_h, bias)
            self.W_o = nn.Linear(d_v + d_h, d_h)
        else:
            self.W_i = nn.Linear(d_v, d_h, bias)
            self.W_h = nn.Linear(d_e + d_h, d_h, bias)
            self.W_o = nn.Linear(d_v + d_h, d_h)

        if atom_descriptors == "descriptor":
            self.__output_dim += d_vd
            self.fc_vd = nn.Linear(d_h + d_vd, d_h + d_vd)

    @property
    def output_dim(self) -> int:
        return self.__output_dim

    def concatenate_descriptors(self, H_v, F_v_d) -> Tensor:
        """Concatenate the atom descriptors F_v_d onto the hidden representations H_v

        Parameters
        ----------
        H_v : Tensor
            a tensor of shape `V x d_h` containing the hidden representation of each atom
        F_v_d : Tensor
            a tensor of shape `V x d_vd` containing additional descriptors for each atom

        Returns
        -------
        Tensor
            a tensor of shape `V x (d_h + d_vd)` containing the transformed hidden representations

        Raises
        ------
        ValueError
            if `F_v_d` is of incorrect shape
        """
        try:
            H_vd = torch.cat((H_v, F_v_d), 1)
        except RuntimeError:
            raise ValueError(
                "arg `F_v_d` has incorrect shape! "
                f"got: `{' x '.join(map(str, F_v_d.shape))}`. expected: `{len(H_v)} x d_vd`"
            )

        H_v = self.fc_vd(H_vd)  # V x (d_h + d_vd)
        H_v = self.dropout(H_v)  # V x (d_h + d_vd)

        return H_v

    def readout(self, H_v: Tensor, sizes: Iterable[int]) -> Tensor:
        h_vs = torch.split(H_v, sizes)
        hs = [h_v.sum(0) if size > 0 else self.cached_zero_vector for h_v, size in zip(h_vs, sizes)]

        if self.aggregation == "mean":
            hs = [h / size if size > 0 else h for h, size in zip(hs, sizes)]
        elif self.aggregation == "norm":
            hs = [h / self.aggregation_norm if size > 0 else h for h, size in zip(hs, sizes)]
        else:
            raise RuntimeError

        return torch.stack(hs, 0)

    @abstractmethod
    def forward(
        self,
        X_v: Tensor,
        X_e: Tensor,
        a2b: Tensor,
        b2a: Tensor,
        b2revb: Tensor,
        a_scope: Iterable[Tuple],
        b_scope: Iterable[Tuple],
        a2a: Optional[Tensor] = None,
        X_v_d: Optional[Tensor] = None,
    ) -> Tensor:
        """Encode a batch of molecular graphs.

        Parameters
        ----------
        X_v : Tensor
            a tensor of shape `V x d_v` containg atom ("vertex") features, where `V` is the number
            of atoms in the batch + 1 and `d_v` is the dimension of the atom features
        X_e : Tensor
            a tensor of shape `E x d_e` containg bond ("edge") features, where `E` is two times the
            number of directed edges in the batch and `d_e` is the dimension of the edge features
        a2b : Tensor
            a mapping from atom index to incoming bond indices
        b2a : Tensor
            a mapping from bond index to the index of the atom the bond is coming from
        b2revb : Tensor
            mapping from bond index to the index of the reverse bond
        a_scope : Iterable[Tuple]
            a list of tuples containing (start_index, num_atoms) for each molecule in the batch
        b_scope : Iterable[Tuple]
            TODO
        X_v_d : Optional[Tensor]
            an optional tensor of shape `V x d_vd` containing additional descriptors for each atom
            in the batch. These will be concatenated to the learned atomic descriptors and
            transformed before the readout phase. NOTE: recall that `V` is equal to `num_atoms` + 1
            so if provided, this tensor must be 0-padded in the 0th row.

        Returns
        -------
        Tensor
            a tensor of shape `B x d_h` or `B x (d_h + d_vd)` containing the encoding of each
            molecule in the batch, depending on whether additional atom descriptors were provided
        """
