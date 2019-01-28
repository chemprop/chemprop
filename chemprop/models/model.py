from argparse import Namespace
from functools import partial
from typing import Dict

import torch.nn as nn
import torch.nn.functional as F

from .mpn import MPN
from chemprop.nn_utils import get_activation_function, initialize_weights, MayrDropout, MayrLinear, MAMLLinear


class MoleculeModel(nn.Module):
    def __init__(self):
        super(MoleculeModel, self).__init__()
    
    def create_encoder(self, args: Namespace, params: Dict[str, nn.Parameter] = None):
        self.encoder = MPN(args, params=params)

    def create_ffn(self, args: Namespace, params: Dict[str, nn.Parameter] = None):
        output_size = args.num_tasks

        first_linear_dim = args.hidden_size
        if args.use_input_features:
            first_linear_dim += args.features_dim
        
        drop_layer = lambda p: nn.Dropout(p)

        def linear_layer(input_dim: int, output_dim: int, p: float, idx: int):
            return nn.Linear(input_dim, output_dim)

        # Create FFN layers
        idx = 1
        if args.ffn_num_layers == 1:
            ffn = [
                drop_layer(args.ffn_input_dropout),
                linear_layer(first_linear_dim, args.output_size, args.ffn_input_dropout, idx)
            ]
        else:
            ffn = [
                drop_layer(args.ffn_input_dropout),
                linear_layer(first_linear_dim, args.ffn_hidden_size, args.ffn_input_dropout, idx)
            ]
            for _ in range(args.ffn_num_layers - 2):
                idx += 3
                ffn.extend([
                    get_activation_function(args.activation),
                    drop_layer(args.ffn_dropout),
                    linear_layer(args.ffn_hidden_size, args.ffn_hidden_size, args.ffn_dropout, idx),
                ])
            idx += 3
            ffn.extend([
                get_activation_function(args.activation),
                drop_layer(args.ffn_dropout),
                linear_layer(args.ffn_hidden_size, args.output_size, args.ffn_dropout, idx),
            ])

        # Classification
        if args.dataset_type == 'classification':
            ffn.append(nn.Sigmoid())

        # Combined model
        self.ffn = nn.Sequential(*ffn)

    def forward(self, *input):
        return self.ffn(self.encoder(*input))


def build_model(args: Namespace, params: Dict[str, nn.Parameter] = None) -> nn.Module:
    """
    Builds a message passing neural network including final linear layers and initializes parameters.

    :param args: Arguments.
    :param params: Parameters to use instead of creating parameters.
    :return: An nn.Module containing the MPN encoder along with final linear layers with parameters initialized.
    """
    output_size = args.num_tasks
    args.output_size = output_size

    model = MoleculeModel()
    model.create_encoder(args, params=params)
    model.create_ffn(args, params=params)

    initialize_weights(model, args)

    return model
