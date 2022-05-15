from __future__ import annotations

from copy import deepcopy
from typing import Iterable

import torch
from torch import Tensor, nn

from chemprop.models.v2.encoders.base import MPNEncoder, MoleculeEncoder
from chemprop.models.v2.encoders.molecule import build_molecule_encoder
from chemprop.nn_utils import weight_init


class ReactionEncoder(MPNEncoder):
    def __init__(
        self, encoder: MoleculeEncoder, n_mols: int, shared: bool = False, reinitialize: bool = True
    ):
        super().__init__()

        self.n_mols = n_mols
        if shared:
            self.encoders = nn.ModuleList([encoder for _ in range(n_mols)])
        else:
            self.encoders = nn.ModuleList([deepcopy(encoder) for _ in range(n_mols)])
            # if reinitialize:
            #     [encoder.apply(weight_init) for encoder in self.encoders]

        self.__output_dim = sum(encoder.output_dim for encoder in self.encoders)
    
    @property
    def output_dim(self) -> int:
        self.__output_dim

    def forward(self, reactant_batches: Iterable[Iterable]) -> Tensor:
        """Encode the reactant_batch

        Parameters
        ----------
        reactant_batch : Iterable[Iterable]
            an Iterable of length `n` containing inputs to a MoleculeEncoder, where `n` is the
            number of molecules in each reaction (== `self.n_mols`). I.e., to encode a batch of 
            3-component reactions, the 0th entry of `reactant_batches` will be the batched inputs of
            the 0th component of each reaction, the 1st entry will be the 1st component, etc. In 
            other words, if you were to (hypothetically) encode a batch of single component 
            reactions (i.e., a batch of molecules), the only difference between a ReactionEncoder 
            and a MoleculeEncoder would be the call signature:

            >>> inputs = X_v, X_e, a2b, ..., X_v_d
            >>> mol_enc: MoleculeEncoder
            >>> rxn_enc: ReactionEncoder
            >>> H_mol = mol_enc(*inputs)
            >>> H_rxn = rxn_enc([inputs])
            >>> H_mol.shape == H_rxn.shape
            True

        Returns
        -------
        Tensor
            a Tensor of shape `b x d_h` containing the reaction encodings, where `b` is the number 
            of reactions in the batch, and `d_h` is equal to `output_dim`
            (== `self.n_mols x self.encoders[0].output_dim`)
        """
        Hs = [encoder(*inputs) for encoder, inputs in zip(self.encoders, reactant_batches)]
        H = torch.cat(Hs, 1)
    
        return H

    @classmethod
    def from_mol_encoder_args(
        cls, n_mols: int, shared: bool = False, reinitialize: bool = True, *args, **kwargs
    ) -> ReactionEncoder:
        encoder = build_molecule_encoder(*args, **kwargs)

        return cls(encoder, n_mols, shared, reinitialize)