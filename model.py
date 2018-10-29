from argparse import Namespace

import torch.nn as nn

from jtnn import JTNN
from mpn import MPN, GAN
from moe import MOE
from nn_utils import get_activation_function, initialize_weights


class MoleculeModel(nn.Module):
    def __init__(self, encoder: nn.Module, ffn: nn.Sequential):
        super(MoleculeModel, self).__init__()
        self.encoder = encoder
        self.ffn = ffn

    def forward(self, *input):
        return self.ffn(self.encoder(*input))


def build_model(args: Namespace) -> nn.Module:
    """
    Builds a message passing neural network including final linear layers and initializes parameters.

    :param args: Arguments.
    :return: An nn.Module containing the MPN encoder along with final linear layers with parameters initialized.
    """
    # Regression with binning
    if args.dataset_type == 'regression_with_binning':
        output_size = args.num_bins * args.num_tasks
    else:
        output_size = args.num_tasks

    # JTNN
    if args.jtnn:
        encoder = JTNN(args)
    else:
        encoder = MPN(args)

    # Learning virtual edges
    if args.learn_virtual_edges:
        args.lve_model = encoder.encoder  # to make this accessible during featurization, to select virtual edges

    if args.moe:
        model = MOE(args)
        initialize_weights(model)

        return model

    # Additional features
    if args.features_only:
        first_linear_dim = args.features_dim
    else:
        first_linear_dim = args.hidden_size * (1 + args.jtnn)
        if args.use_input_features:
            first_linear_dim += args.features_dim

    # Create FFN layers
    if args.ffn_num_layers == 1:
        ffn = [
            nn.Dropout(args.ffn_input_dropout),
            nn.Linear(first_linear_dim, output_size)
        ]
    else:
        ffn = [
            nn.Dropout(args.ffn_input_dropout),
            nn.Linear(first_linear_dim, args.ffn_hidden_dim)
        ]
        for _ in range(args.ffn_num_layers - 2):
            ffn.extend([
                get_activation_function(args.activation),
                nn.Dropout(args.ffn_dropout),
                nn.Linear(args.ffn_hidden_dim, args.ffn_hidden_dim),
            ])
        ffn.extend([
            get_activation_function(args.activation),
            nn.Dropout(args.ffn_dropout),
            nn.Linear(args.ffn_hidden_dim, output_size),
        ])
    
    # Classification
    if args.dataset_type == 'classification':
        ffn.append(nn.Sigmoid())

    # Combined model
    ffn = nn.Sequential(*ffn)
    model = MoleculeModel(encoder, ffn)

    if args.adversarial:
        args.output_size = output_size
        model = GAN(args, model)

    initialize_weights(model)

    return model
