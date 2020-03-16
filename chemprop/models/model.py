from argparse import Namespace

import torch
import torch.nn as nn

from .mpn import MPN
from chemprop.nn_utils import get_activation_function, initialize_weights


class MoleculeModel(nn.Module):
    """A MoleculeModel is a model which contains a message passing network following by feed-forward layers."""

    def __init__(self, output_raw: bool):
        """
        Initializes the MoleculeModel.

        :param raw_score: Whether the model should apply activation to output.
        """
        super(MoleculeModel, self).__init__()

        self.activation = nn.Identity()
        if output_raw:
            self.activation = nn.Sigmoid()

    def create_encoder(self, args: Namespace):
        """
        Creates the paired message passing encoders for the model.

        :param args: Arguments.
        """
        self.drug_encoder = MPN(args) if not args.cmpd_only else None
        self.cmpd_encoder = MPN(args) if not args.drug_only else None

    def create_ffn(self, args: Namespace):
        """
        Creates the feed-forward network for the model.

        :param args: Arguments.
        """
        self.multiclass = args.dataset_type == 'multiclass'
        self.ops = args.ops

        if self.multiclass:
            self.num_classes = args.multiclass_num_classes
        if args.features_only:
            first_linear_dim = args.features_size
        else:
            first_linear_dim = args.hidden_size*2
            if args.use_input_features:
                first_linear_dim += args.features_dim

        if args.drug_only or args.cmpd_only or self.ops != 'concat':
            first_linear_dim = int(first_linear_dim/2)

        dropout = nn.Dropout(args.dropout)
        activation = get_activation_function(args.activation)

        # Create FFN layers
        if args.ffn_num_layers == 1:
            ffn = [
                dropout,
                nn.Linear(first_linear_dim, args.output_size)
            ]
        else:
            ffn = [
                dropout,
                nn.Linear(first_linear_dim, args.ffn_hidden_size)
            ]
            for _ in range(args.ffn_num_layers - 2):
                ffn.extend([
                    activation,
                    dropout,
                    nn.Linear(args.ffn_hidden_size, args.ffn_hidden_size),
                ])
            ffn.extend([
                activation,
                dropout,
                nn.Linear(args.ffn_hidden_size, args.output_size),
            ])

        # Create FFN model
        self.ffn = nn.Sequential(*ffn)

    def forward(self, *input):
        """
        Runs the MoleculeModel on input.

        :param input: Input.
        :return: The output of the MoleculeModel.
        """
        smiles, feats = input

        newInput = []
        if self.drug_encoder:
            learned_drug = self.drug_encoder([x[0] for x in smiles], [x[0] for x in feats])
            newInput.append(learned_drug)
        if self.cmpd_encoder:
            learned_cmpd = self.cmpd_encoder([x[1] for x in smiles], [x[1] for x in feats])
            newInput.append(learned_cmpd)

        assert len(newInput) != 0

        if len(newInput) > 1:
            if self.ops == 'plus':
                newInput = newInput[0] + newInput[1]
            elif self.ops == 'minus':
                newInput = newInput[0] - newInput[1]
            else:
                newInput = torch.cat(newInput, dim=1)
        else:
            newInput = newInput[0]

        output = self.ffn(newInput)

        # Don't apply sigmoid during training b/c using BCEWithLogitsLoss
        if not self.training:
            output = self.activation(output)

        return output


def build_model(args: Namespace) -> nn.Module:
    """
    Builds a MoleculeModel, which is a message passing neural network + feed-forward layers.

    :param args: Arguments.
    :return: A MoleculeModel containing the MPN encoder along with final linear layers with parameters initialized.
    """
    output_size = args.num_tasks
    args.output_size = output_size
    if args.dataset_type == 'multiclass':
        raise NotImplementedError

    model = MoleculeModel(output_raw=args.output_raw)
    model.create_encoder(args)
    model.create_ffn(args)

    initialize_weights(model)

    return model
