from argparse import Namespace

import torch.nn as nn

from .gan import GAN
from .jtnn import JTNN
from .moe import MOE
from .mpn import MPN, MPNEncoder
from chemprop.features import get_atom_fdim, get_bond_fdim
from chemprop.nn_utils import get_activation_function, initialize_weights, MayrDropout, MayrLinear


class MoleculeModel(nn.Module):
    def __init__(self):
        super(MoleculeModel, self).__init__()
    
    def create_encoder(self, args):
        if args.jtnn:
            self.encoder = JTNN(args)
        elif args.dataset_type == 'bert_pretraining':
            atom_fdim = get_atom_fdim(args)
            bond_fdim = atom_fdim + get_bond_fdim(args)
            self.encoder = MPNEncoder(args, atom_fdim, bond_fdim)
        else:
            self.encoder = MPN(args)

    def create_ffn(self, args):
        if args.dataset_type == 'bert_pretraining':
            self.ffn = lambda x: x
            return

        if args.dataset_type == 'regression_with_binning':
            output_size = args.num_bins * args.num_tasks
        elif args.dataset_type == 'unsupervised':
            output_size = args.unsupervised_n_clusters
        else:
            output_size = args.num_tasks

        # Additional features
        if args.features_only:
            first_linear_dim = args.features_dim
        else:
            first_linear_dim = args.hidden_size * (1 + args.jtnn)
            if args.use_input_features:
                first_linear_dim += args.features_dim
        
        if args.mayr_layers:
            drop_layer = lambda p: MayrDropout(p)
            linear_layer = lambda input_dim, output_dim, p: MayrLinear(input_dim, output_dim, p)
        else:
            drop_layer = lambda p: nn.Dropout(p)
            linear_layer = lambda input_dim, output_dim, p: nn.Linear(input_dim, output_dim)

        # Create FFN layers
        if args.ffn_num_layers == 1:
            ffn = [
                drop_layer(args.ffn_input_dropout),
                linear_layer(first_linear_dim, output_size, args.ffn_input_dropout)
            ]
        else:
            ffn = [
                drop_layer(args.ffn_input_dropout),
                linear_layer(first_linear_dim, args.ffn_hidden_size, args.ffn_input_dropout)
            ]
            for _ in range(args.ffn_num_layers - 2):
                ffn.extend([
                    get_activation_function(args.activation),
                    drop_layer(args.ffn_dropout),
                    linear_layer(args.ffn_hidden_size, args.ffn_hidden_size, args.ffn_dropout),
                ])
            ffn.extend([
                get_activation_function(args.activation),
                drop_layer(args.ffn_dropout),
                linear_layer(args.ffn_hidden_size, output_size, args.ffn_dropout),
            ])

        # Classification
        if args.dataset_type == 'classification':
            ffn.append(nn.Sigmoid())

        # Combined model
        ffn = nn.Sequential(*ffn)

        self.ffn = ffn

    def forward(self, *input):
        return self.ffn(self.encoder(*input))


def build_model(args: Namespace) -> nn.Module:
    """
    Builds a message passing neural network including final linear layers and initializes parameters.

    :param args: Arguments.
    :return: An nn.Module containing the MPN encoder along with final linear layers with parameters initialized.
    """
    # Learning virtual edges
    if args.learn_virtual_edges:
        args.lve_model = encoder.encoder  # to make this accessible during featurization, to select virtual edges

    if args.moe:
        model = MOE(args)
        if args.adversarial:
            args.output_size = output_size
            model = GAN(args, prediction_model=model, encoder=model.encoder)
        initialize_weights(model)

        return model

    model = MoleculeModel()
    model.create_encoder(args)
    model.create_ffn(args)

    if args.adversarial:
        args.output_size = output_size
        model = GAN(args, prediction_model=model, encoder=model.encoder)

    initialize_weights(model)

    return model
