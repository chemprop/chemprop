from argparse import Namespace

import torch.nn as nn

from jtnn import JTNN
from mpn import MPN, GAN


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

    # Additional features
    if args.features_only:
        first_linear_dim = args.features_dim
    else:
        first_linear_dim = args.hidden_size * (1 + args.jtnn)
        if args.features:
            first_linear_dim += args.features_dim

    if args.features_only or args.more_ffn_capacity:
        modules = [
            encoder,
            nn.Dropout(args.ffn_input_dropout),
            nn.Linear(first_linear_dim, args.ffn_hidden_dim),
            nn.ReLU(),
            nn.Dropout(args.ffn_dropout),
            nn.Linear(args.ffn_hidden_dim, args.ffn_hidden_dim),
            nn.ReLU(),
            nn.Dropout(args.ffn_dropout),
            nn.Linear(args.ffn_hidden_dim, args.ffn_hidden_dim),
            nn.ReLU(),
            nn.Dropout(args.ffn_dropout),
            nn.Linear(args.ffn_hidden_dim, output_size)
        ]
    else:
        modules = [
            encoder,
            nn.Linear(first_linear_dim, args.hidden_size),
            nn.ReLU(),
            nn.Linear(args.hidden_size, output_size)
        ]

    # Classification
    if args.dataset_type == 'classification':
        modules.append(nn.Sigmoid())

    # Combined model
    model = nn.Sequential(*modules)

    if args.adversarial:
        args.output_size = output_size
        model = GAN(args, model)

    # Initialize weights
    for param in model.parameters():
        if param.dim() == 1:
            nn.init.constant_(param, 0)
        else:
            nn.init.xavier_normal_(param)

    return model
