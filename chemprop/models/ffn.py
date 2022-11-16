from typing import List, Tuple, Optional

import torch
import torch.nn as nn

from chemprop.nn_utils import get_activation_function

class MultiReadout(nn.Module):
    """A :class:`MultiReadout` contains a list of FFN for each atom/bond targets prediction."""

    def __init__(
        self,
        atom_features_size: int,
        bond_features_size: int,
        atom_hidden_size: int,
        bond_hidden_size: int,
        num_layers: int,
        output_size: int,
        dropout: float,
        activation: str,
        atom_constraints: List[bool] = None,
        bond_constraints: List[bool] = None,
        shared_ffn: bool = True,
        weights_ffn_num_layers: int = 2,
    ):
        """
        :param atom_features_size: Dimensionality of input atomic features.
        :param bond_features_size: Dimensionality of input bond features.
        :param atom_hidden_size: Dimensionality of atomic hidden layers.
        :param bond_hidden_size: Dimensionality of bond hidden layers.
        :param num_layers: Number of layers in FFN.
        :param output_size: The size of output.
        :param dropout: Dropout probability.
        :param activation: Activation function.
        :param atom_constraints: A list of booleans indicating whether constraints applied to output of atomic properties.
        :param bond_constraints: A list of booleans indicating whether constraints applied to output of bond properties.
        :param shared_ffn: Whether to share weights in the ffn between different atom tasks and bond tasks.
        :param weights_ffn_num_layers: Number of layers in FFN for determining weights used to correct the constrained targets.
        """
        super().__init__()

        if num_layers > 1 and shared_ffn:
            self.atom_ffn_base = nn.Sequential(
                build_ffn(
                    first_linear_dim=atom_features_size,
                    hidden_size=atom_hidden_size,
                    num_layers=num_layers - 1,
                    output_size=atom_hidden_size,
                    dropout=dropout,
                    activation=activation,
                ),
                get_activation_function(activation),
            )
            self.bond_ffn_base = nn.Sequential(
                build_ffn(
                    first_linear_dim=2*bond_features_size,
                    hidden_size=bond_hidden_size,
                    num_layers=num_layers - 1,
                    output_size=bond_hidden_size,
                    dropout=dropout,
                    activation=activation,
                ),
                get_activation_function(activation),
            )
        else:
            self.atom_ffn_base = None
            self.bond_ffn_base = None

        atom_ffn_params = {
            "features_size": atom_features_size,
            "hidden_size": atom_hidden_size,
            "num_layers": num_layers,
            "output_size": output_size,
            "dropout": dropout,
            "activation": activation,  
            "ffn_base": self.atom_ffn_base,
            "ffn_type": "atom",
        }

        bond_ffn_params = {
            "features_size": 2*bond_features_size,
            "hidden_size": bond_hidden_size,
            "num_layers": num_layers,
            "output_size": output_size,
            "dropout": dropout,
            "activation": activation,  
            "ffn_base": self.bond_ffn_base,
            "ffn_type": "bond",
        }

        ffn_list = []
        for constraint in atom_constraints:
            if constraint:
                ffn_list.append(
                    FFNAtten(
                        weights_ffn_num_layers=weights_ffn_num_layers,
                        **atom_ffn_params,
                    )
                )
            else:
                ffn_list.append(FFN(**atom_ffn_params))

        for constraint in bond_constraints:
            if constraint:
                ffn_list.append(
                    FFNAtten(
                        weights_ffn_num_layers=weights_ffn_num_layers,
                        **bond_ffn_params,
                    )
                )
            else:
                ffn_list.append(FFN(**bond_ffn_params))

        self.ffn_list = nn.ModuleList(ffn_list)

    def forward(
        self,
        input: Tuple[torch.Tensor, List, torch.Tensor, List, torch.Tensor],
        constraints_batch: List[torch.Tensor],
        bond_types_batch: List[torch.Tensor],
    ) -> List[torch.Tensor]:
        """
        Runs the :class:`MultiReadout` on input.
        :param input: A tuple of atomic and bond information of each molecule.
        :param constraints_batch: A list of PyTorch tensors which applies constraint on atomic/bond properties.
        :param bond_types_batch: A list of PyTorch tensors storing bond types of each bond determined by RDKit molecules.
        :return: The output of the :class:`MultiReadout`, a list of PyTorch tensors which contains atomic/bond properties prediction.
        """
        results = []
        for i, ffn in enumerate(self.ffn_list):
            if isinstance(ffn, FFNAtten):
                results.append(ffn(input, constraints_batch[i], bond_types_batch[i]))
            else:
                results.append(ffn(input, bond_types_batch[i]))
        return results


class FFN(nn.Module):
    """
    A :class:`FFN` is a multiple feed forward neural networks (FFN) to predict
    the atom/bond descriptors without applying constraint on output.
    """

    def __init__(
        self,
        features_size: int,
        hidden_size: int,
        num_layers: int,
        output_size: int,
        dropout: float,
        activation: str,
        ffn_base: Optional[nn.Module] = None,
        ffn_type: str = "atom",
    ):
        """
        :param features_size: Dimensionality of input features.
        :param hidden_size: Dimensionality of hidden layers.
        :param num_layers: Number of layers in FFN.
        :param output_size: The size of output.
        :param dropout: Dropout probability.
        :param activation: Activation function.
        :param ffn_base: The shared base layers (all but the last) of the FFN between tasks.
        :param ffn_type: The type of target (atom or bond).
        """
        super().__init__()

        base_output_size = features_size if num_layers == 1 else hidden_size

        if ffn_base:
            self.ffn = ffn_base
        else:
            if num_layers > 1:
                self.ffn = nn.Sequential(
                    build_ffn(
                        first_linear_dim=features_size,
                        hidden_size=hidden_size,
                        num_layers=num_layers - 1,
                        output_size=hidden_size,
                        dropout=dropout,
                        activation=activation,
                    ),
                    get_activation_function(activation),
                )
            else:
                self.ffn = nn.Identity()
        self.ffn_readout = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(base_output_size, output_size),
        )
        self.ffn_type = ffn_type

    def calc_hidden(
        self,
        input: Tuple[torch.Tensor, List, torch.Tensor, List, torch.Tensor],
    ) -> torch.Tensor:
        """
        Calculate the hidden representation for each atom or bond in a molecule.
        :param input: A tuple of atom and bond informations of each molecule.
        :return: The hidden representation for each atom or bond in a molecule.
        """
        a_hidden, _, b_hidden, _, b2br = input

        if self.ffn_type == "atom":
            hidden = a_hidden
        elif self.ffn_type == "bond":
            forward_bond = b_hidden[b2br[:, 0]]
            backward_bond = b_hidden[b2br[:, 1]]
        else:
            raise RuntimeError(f"Unrecognized ffn_type of {self.ffn_type}.")

        if self.ffn_type == "atom":
            output_hidden = self.ffn(hidden)
        elif self.ffn_type == "bond":
            b_hidden_1 = torch.cat([forward_bond, backward_bond], dim=1)
            b_hidden_2 = torch.cat([backward_bond, forward_bond], dim=1)
            output_1 = self.ffn(b_hidden_1)
            output_2 = self.ffn(b_hidden_2)
            output_hidden = (output_1 + output_2) / 2

        return output_hidden

    def readout(
        self,
        input: torch.Tensor,
        bond_types: torch.Tensor,
    ) -> torch.Tensor:
        """
        Runs the :class:`FFN` on input hidden representation.
        :param input: The hidden representation for each atom or bond in a molecule.
        :param bond_types: A PyTorch tensor storing bond types of each bond determined by RDKit molecules.
        :return: The output of the :class:`FFN`, a PyTorch tensor containing a list of property predictions.
        """
        output = self.ffn_readout(input)
        if self.ffn_type == "atom":
            output = output[1:]  # remove the first one which is zero padding
        elif self.ffn_type == "bond" and bond_types is not None:
            output = output + bond_types.reshape(-1, 1)

        return output

    def forward(
        self,
        input: Tuple[torch.Tensor, List, torch.Tensor, List, torch.Tensor],
        bond_types: torch.Tensor,
    ) -> torch.Tensor:
        """
        Runs the :class:`FFN` on input.
        :param input: A tuple of atom and bond informations of each molecule.
        :param constraints: A PyTorch tensor which applies constraint on atomic/bond properties.
        :param bond_types: A PyTorch tensor storing bond types of each bond determined by RDKit molecules.
        :return: The output of the :class:`FFN`, a PyTorch tensor containing a list of property predictions.
        """
        output_hidden = self.calc_hidden(input)
        output = self.readout(output_hidden, bond_types)

        return output

class FFNAtten(FFN):
    """
    A :class:`FFNAtten` is a multiple feed forward neural networks (FFN) to predict
    the atom/bond descriptors with applying constraint on output. An attention-based
    constraint is used. This method is from `Regio-selectivity prediction with a
    machine-learned reaction representation and on-the-fly quantum mechanical descriptors
    <https://pubs.rsc.org/en/content/articlelanding/2021/sc/d0sc04823b>`_, section 2.2.
    """

    def __init__(
        self,
        features_size: int,
        hidden_size: int,
        num_layers: int,
        output_size: int,
        dropout: float,
        activation: str,
        ffn_base: Optional[nn.Module] = None,
        ffn_type: str = "atom",
        weights_ffn_num_layers: int = 2,
    ):
        """
        :param features_size: Dimensionality of input features.
        :param hidden_size: Dimensionality of hidden layers.
        :param num_layers: Number of layers in FFN.
        :param output_size: The size of output.
        :param dropout: Dropout probability.
        :param activation: Activation function.
        :param ffn_base: The shared base layers (all but the last) of the FFN between tasks.
        :param ffn_type: The type of target (atom or bond).
        :param weights_ffn_num_layers: Number of layers in FFN for determining weights used to correct the constrained targets.
        """
        super().__init__(
            features_size,
            hidden_size,
            num_layers,
            output_size,
            dropout,
            activation,
            ffn_base,
            ffn_type,
        )

        base_output_size = features_size if num_layers == 1 else hidden_size

        self.weights_readout = build_ffn(
            first_linear_dim=base_output_size,
            hidden_size=hidden_size,
            output_size=1,
            num_layers=weights_ffn_num_layers,
            dropout=dropout,
            activation=activation,
        )

    def readout(
        self,
        input: torch.Tensor,
        scope: List[Tuple],
        constraints: torch.Tensor,
        bond_types: torch.Tensor,
    ) -> torch.Tensor:
        """
        Runs the :class:`FFNAtten` on hidden representation.
        :param input: The hidden representation for each atom or bond in a molecule.
        :param scope: A list of tuples indicating the start and end atom/bond indices for each molecule.
        :param constraints: A PyTorch tensor which applies constraint on atomic/bond properties.
        :param bond_types: A PyTorch tensor storing bond types of each bond determined by RDKit molecules.
        :return: The output of the :class:`FFN`, a PyTorch tensor containing a list of property predictions.
        """
        output = self.ffn_readout(input)
        if self.ffn_type == "bond" and bond_types is not None:
            output = output + bond_types.reshape(-1, 1)

        W_a = self.weights_readout(input)
        constrained_output = []
        for i, (start, size) in enumerate(scope):
            if size == 0:
                continue
            else:
                q_i = output[start:start+size]
                w_i = W_a[start:start+size].softmax(0)
                Q = constraints[i]
                q_f = q_i + w_i * (Q - q_i.sum())
                constrained_output.append(q_f)

        output = torch.cat(constrained_output, dim=0)

        return output

    def forward(
        self,
        input: Tuple[torch.Tensor, List, torch.Tensor, List, torch.Tensor],
        constraints: torch.Tensor,
        bond_types: torch.Tensor,
    ) -> torch.Tensor:
        """
        Runs the :class:`FFNAtten` on input.
        :param input: A tuple of atom and bond informations of each molecule.
        :param constraints: A PyTorch tensor which applies constraint on atomic/bond properties.
        :param bond_types: A PyTorch tensor storing bond types of each bond determined by RDKit molecules.
        :return: The output of the :class:`FFNAtten`, a PyTorch tensor containing a list of property predictions.
        """
        output_hidden = self.calc_hidden(input)
        _, a_scope, _, b_scope, _ = input
        if self.ffn_type == "atom":
            scope = a_scope
        elif self.ffn_type == "bond":
            scope = [((start - 1) // 2, size // 2) for start, size in b_scope]
        output = self.readout(output_hidden, scope, constraints, bond_types)

        return output


class Exp(nn.Module):
    def forward(self, x):
        return x.exp()

def build_ffn(
    first_linear_dim: int,
    hidden_size: int,
    num_layers: int,
    output_size: int,
    dropout: float,
    activation: str,
    dataset_type: str = None,
    spectra_activation: str = None,
) -> nn.Sequential:
    """
    Returns an `nn.Sequential` object of FFN layers.

    :param first_linear_dim: Dimensionality of fisrt layer.
    :param hidden_size: Dimensionality of hidden layers.
    :param num_layers: Number of layers in FFN.
    :param output_size: The size of output.
    :param dropout: Dropout probability.
    :param activation: Activation function.
    :param dataset_type: Type of dataset.
    :param spectra_activation: Activation function used in dataset_type spectra training to constrain outputs to be positive.
    """
    activation = get_activation_function(activation)

    if num_layers == 1:
        layers = [
            nn.Dropout(dropout),
            nn.Linear(first_linear_dim, output_size)
        ]
    else:
        layers = [
            nn.Dropout(dropout),
            nn.Linear(first_linear_dim, hidden_size)
        ]
        for _ in range(num_layers - 2):
            layers.extend([
                activation,
                nn.Dropout(dropout),
                nn.Linear(hidden_size, hidden_size),
            ])
        layers.extend([
            activation,
            nn.Dropout(dropout),
            nn.Linear(hidden_size, output_size),
        ])

    # If spectra model, also include spectra activation
    if dataset_type == "spectra":
        spectra_activation = nn.Softplus() if spectra_activation == "softplus" else Exp()
        layers.append(spectra_activation)

    return nn.Sequential(*layers)
