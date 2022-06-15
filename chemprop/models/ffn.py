from typing import List, Tuple

import torch
import torch.nn as nn
import re


class AttrProxy(object):
    """Translates index lookups into attribute lookups."""

    def __init__(self, module, prefix):
        self.module = module
        self.prefix = prefix

    def __len__(self):
        return len([x for x in self.module.__dict__['_modules'].keys() if re.match(f'{self.prefix}\d+', x)])

    def __getitem__(self, item):
        if item >= len(self):
            raise IndexError
        return getattr(self.module, self.prefix + str(item))


class MultiReadout(nn.Module):
    """A fake list of FFNs for reading out as suggested in
    https://discuss.pytorch.org/t/list-of-nn-module-in-a-nn-module/219/3"""

    def __init__(
        self,
        atom_features_size: int,
        bond_features_size: int,
        atom_hidden_size: int,
        bond_hidden_size: int,
        num_layers: int,
        output_size: int,
        dropout: nn.Module,
        activation: nn.Module,
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
        super(MultiReadout, self).__init__()
        self.shared_ffn = shared_ffn

        if num_layers > 1 and shared_ffn:
            self.atom_ffn_base = nn.Sequential(
                DenseLayers(
                    first_linear_dim=atom_features_size,
                    hidden_size=atom_hidden_size,
                    num_layers=num_layers - 1,
                    output_size=atom_hidden_size,
                    dropout=dropout,
                    activation=activation,
                ),
                activation,
            )
            self.bond_ffn_base = nn.Sequential(
                DenseLayers(
                    first_linear_dim=2*bond_features_size,
                    hidden_size=bond_hidden_size,
                    num_layers=num_layers - 1,
                    output_size=bond_hidden_size,
                    dropout=dropout,
                    activation=activation,
                ),
                activation,
            )
        else:
            self.atom_ffn_base = nn.Identity()
            self.bond_ffn_base = nn.Identity()

        ind = 0

        for constraint in atom_constraints:
            self.add_module(
                f"readout_{ind}",
                FFNAtten(
                    features_size=atom_features_size,
                    hidden_size=atom_hidden_size,
                    num_layers=num_layers,
                    output_size=output_size,
                    dropout=dropout,
                    activation=activation,
                    ffn_base=self.atom_ffn_base,
                    constraint=constraint,
                    ffn_type="atom",
                    shared_ffn=shared_ffn,
                    weights_ffn_num_layers=weights_ffn_num_layers,
                ),
            )
            ind += 1

        for constraint in bond_constraints:
            self.add_module(
                f"readout_{ind}",
                FFNAtten(
                    features_size=2*bond_features_size,
                    hidden_size=bond_hidden_size,
                    num_layers=num_layers,
                    output_size=output_size,
                    dropout=dropout,
                    activation=activation,
                    ffn_base=self.bond_ffn_base,
                    constraint=constraint,
                    ffn_type="bond",
                    shared_ffn=shared_ffn,
                    weights_ffn_num_layers=weights_ffn_num_layers,
                ),
            )
            ind += 1

        self.ffn_list = AttrProxy(self, "readout_")

    def forward(
        self,
        input: Tuple[torch.tensor, List, torch.tensor, List, torch.tensor],
        constraints_batch: List[torch.tensor],
        bond_types_batch: List[torch.FloatTensor],
    ) -> List[torch.tensor]:
        """
        Runs the :class:`MultiReadout` on input.
        :param input: A tuple of atomic and bond information of each molecule.
        :param constraints_batch: A list of PyTorch tensors which applies constraint on atomic/bond properties.
        :param bond_types_batch: A list of PyTorch tensors storing bond types of each bond determined by RDKit molecules.
        :return: The output of the :class:`MultiReadout`, a list of PyTorch tensors which ontains atomic/bond properties prediction.
        """
        return [ffn(input, constraints_batch[i], bond_types_batch[i]) for i, ffn in enumerate(self.ffn_list)]


class FFNAtten(nn.Module):
    """
    A :class:`FFNAtten` is a multiple feed forward neural networks (NN) to predict
    the atom/bond descriptors. For constrained descriptors, an attention-based
    constraint is applied. This metthod is from `Regio-selectivity prediction with a
    machinelearned reaction representation and on-the-fly quantum mechanical descriptors
    <https://pubs.rsc.org/en/content/articlelanding/2021/sc/d0sc04823b>`_, section 2.2.
    """

    def __init__(
        self,
        features_size: int,
        hidden_size: int,
        num_layers: int,
        output_size: int,
        dropout: nn.Module,
        activation: nn.Module,
        ffn_base: nn.Module,
        constraint: bool = False,
        ffn_type: str = "atom",
        shared_ffn: bool = True,
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
        :param constraint: Whether to apply constraint to output.
        :param ffn_type: The type of target (atom or bond).
        :param shared_ffn: Whether to share weights in the ffn between different atom tasks and bond tasks.
        :param weights_ffn_num_layers: Number of layers in FFN for determining weights used to correct the constrained targets.
        """
        super(FFNAtten, self).__init__()

        if num_layers == 1:
            base_output_size = features_size
        else:
            base_output_size = hidden_size

        if constraint:
            if shared_ffn:
                self.ffn = ffn_base
            else:
                if num_layers > 1:
                    self.ffn = nn.Sequential(
                        DenseLayers(
                            first_linear_dim=features_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers - 1,
                            output_size=hidden_size,
                            dropout=dropout,
                            activation=activation,
                        ),
                        activation,
                    )
                else:
                    self.ffn = nn.Identity()
            self.ffn_readout = DenseLayers(
                first_linear_dim=base_output_size,
                hidden_size=hidden_size,
                num_layers=1,
                output_size=output_size,
                dropout=dropout,
                activation=activation,
            )
            self.weights_readout = DenseLayers(
                first_linear_dim=base_output_size,
                hidden_size=hidden_size,
                output_size=1,
                num_layers=weights_ffn_num_layers,
                dropout=dropout,
                activation=activation,
            )
        else:
            if shared_ffn:
                self.ffn_readout = nn.Sequential(
                    ffn_base,
                    DenseLayers(
                        first_linear_dim=base_output_size,
                        hidden_size=hidden_size,
                        num_layers=1,
                        output_size=output_size,
                        dropout=dropout,
                        activation=activation,
                    ),
                )
            else:
                self.ffn_readout = DenseLayers(
                    first_linear_dim=features_size,
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                    output_size=output_size,
                    dropout=dropout,
                    activation=activation,
                )
        self.constraint = constraint
        self.ffn_type = ffn_type

    def forward(
        self,
        input: Tuple[torch.tensor, List, torch.tensor, List, torch.tensor],
        constraints: torch.tensor,
        bond_types: torch.tensor,
    ) -> torch.tensor:
        """
        Runs the :class:`FFNAtten` on input.
        :param input: A tuple of atom and bond informations of each molecule.
        :param constraints: A PyTorch tensor which applies constraint on atomic/bond properties.
        :param bond_types: A PyTorch tensor storing bond types of each bond determined by RDKit molecules.
        :return: The output of the :class:`FFNAtten`, a PyTorch tensor containing a list of property predictions.
        """
        a_hidden, a_scope, b_hidden, b_scope, b2br = input

        if self.ffn_type == "atom":
            hidden = a_hidden
            scope = a_scope

        elif self.ffn_type == "bond":
            forward_bond = b_hidden[b2br[:, 0]]
            backward_bond = b_hidden[b2br[:, 1]]
            scope = [((start - 1) // 2, size // 2) for start, size in b_scope]

        if constraints is not None:
            if self.ffn_type == "atom":
                output_hidden = self.ffn(hidden)
                output = self.ffn_readout(output_hidden)
            elif self.ffn_type == "bond":
                forward_bond_hidden = self.ffn(forward_bond)
                backward_bond_hidden = self.ffn(backward_bond)
                output_hidden = forward_bond_hidden.add(backward_bond_hidden)
                output = self.ffn_readout(output_hidden)
                if bond_types is not None:
                    output = output + bond_types.reshape(-1, 1)

            weights = self.weights_readout(output_hidden)
            constrained_output = []
            for i, (start, size) in enumerate(scope):
                if size == 0:
                    continue
                else:
                    cur_weights = weights.narrow(0, start, size)
                    cur_output = output.narrow(0, start, size)

                    cur_weights = torch.nn.Softmax(dim=0)(cur_weights).reshape(-1,)

                    cur_output_sum = cur_output[:, 0].sum()

                    cur_output[:, 0] = cur_output[:, 0] + cur_weights * (constraints[i] - cur_output_sum)
                    constrained_output.append(cur_output)

            output = torch.cat(constrained_output, dim=0)
        else:
            if self.ffn_type == "atom":
                output = self.ffn_readout(hidden)
                output = output[1:]  # remove the first one which is zero padding
            elif self.ffn_type == "bond":
                b_hidden_1 = torch.cat([forward_bond, backward_bond], dim=1)
                b_hidden_2 = torch.cat([backward_bond, forward_bond], dim=1)
                output_1 = self.ffn_readout(b_hidden_1)
                output_2 = self.ffn_readout(b_hidden_2)
                output = (output_1 + output_2) / 2
                if bond_types is not None:
                    output = output + bond_types.reshape(-1, 1)

        return output


class DenseLayers(nn.Module):
    """A :class:`DenseLayers` is a object of dense layers."""

    def __init__(
        self,
        first_linear_dim: int,
        hidden_size: int,
        num_layers: int,
        output_size: int,
        dropout: nn.Module,
        activation: nn.Module,
        dataset_type: str = None,
        spectra_activation: str = None,
    ):
        """
        :param first_linear_dim: Dimensionality of fisrt layer.
        :param hidden_size: Dimensionality of hidden layers.
        :param num_layers: Number of layers in FFN.
        :param output_size: The size of output.
        :param dropout: Dropout probability.
        :param activation: Activation function.
        :param dataset_type: Type of dataset.
        :param spectra_activation: Activation function used in dataset_type spectra training to constrain outputs to be positive.
        """
        super(DenseLayers, self).__init__()
        if num_layers == 1:
            layers = [
                dropout,
                nn.Linear(first_linear_dim, output_size)
            ]
        else:
            layers = [
                dropout,
                nn.Linear(first_linear_dim, hidden_size)
            ]
            for _ in range(num_layers - 2):
                layers.extend([
                    activation,
                    dropout,
                    nn.Linear(hidden_size, hidden_size),
                ])
            layers.extend([
                activation,
                dropout,
                nn.Linear(hidden_size, output_size),
            ])

        # If spectra model, also include spectra activation
        if dataset_type == "spectra":
            if spectra_activation == "softplus":
                spectra_activation = nn.Softplus()
            else:  # default exponential activation which must be made into a custom nn module

                class nn_exp(torch.nn.Module):
                    def __init__(self):
                        super(nn_exp, self).__init__()

                    def forward(self, x):
                        return torch.exp(x)

                spectra_activation = nn_exp()
            layers.append(spectra_activation)

        self.dense_layers = nn.Sequential(*layers)

    def forward(self, input: torch.tensor) -> torch.tensor:
        """
        Runs the :class:`DenseLayers` on input.
        :param input: A PyTorch tensor containing the encoding of each molecule.
        :return: The output of the :class:`DenseLayers`.
        """
        return self.dense_layers(input)
