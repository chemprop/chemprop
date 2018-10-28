from argparse import Namespace

import torch.nn as nn

from jtnn import JTNN
from mpn import MPN, GAN
from moe import MOE

def build_model(args: Namespace) -> nn.Module:
    """
    Builds a message passing neural network including final linear layers and initializes parameters.

    :param args: Arguments.
    :return: An nn.Module containing the MPN encoder along with final linear layers with parameters initialized.
    """
    if args.dataset_type == 'regression_with_binning':
        output_size = args.num_bins * args.num_tasks
    else:
        output_size = args.num_tasks

    if args.jtnn:
        encoder = JTNN(args)
    else:
        encoder = MPN(args)
    if args.learn_virtual_edges:
        args.lve_model = encoder.encoder # to make this accessible during featurization, to select virtual edges
    
    if args.semiF_only:
        first_linear_dim = args.semiF_dim
    else:
        first_linear_dim = args.hidden_size * (1 + args.jtnn)
        if args.semiF_path:
            first_linear_dim += args.semiF_dim
    
    if args.moe:
        model = MOE(args)
    else:
        if args.semiF_only or args.more_ffn_capacity:
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
        if args.dataset_type == 'classification':
            modules.append(nn.Sigmoid())

        model = nn.Sequential(*modules)

        if args.adversarial:
            args.output_size = output_size
            model = GAN(args, model)

    for param in model.parameters():
        if param.dim() == 1:
            nn.init.constant_(param, 0)
        else:
            nn.init.xavier_normal_(param)

    return model
