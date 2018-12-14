from argparse import Namespace
from functools import partial
from typing import Dict

import torch.nn as nn
import torch.nn.functional as F

from .gan import GAN
from .jtnn import JTNN
from .moe import MOE
from .mpn import MPN
from .learned_kernel import LearnedKernel
from chemprop.nn_utils import get_activation_function, initialize_weights, MayrDropout, MayrLinear, MAMLLinear


class MoleculeModel(nn.Module):
    def __init__(self):
        super(MoleculeModel, self).__init__()
    
    def create_encoder(self, args: Namespace, params: Dict[str, nn.Parameter] = None):
        if args.jtnn:
            if params is not None:
                raise ValueError('Setting parameters not yeet supported for JTNN')
            self.encoder = JTNN(args)
        elif args.dataset_type == 'bert_pretraining':
            self.encoder = MPN(args, graph_input=True, params=params)
        else:
            self.encoder = MPN(args, params=params)
        
        if args.freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
        
        if args.gradual_unfreezing:
            self.create_unfreeze_queue(args)

    def create_ffn(self, args: Namespace, params: Dict[str, nn.Parameter] = None):
        # Learning virtual edges
        if args.learn_virtual_edges:
            args.lve_model = self.encoder  # to make this accessible during featurization, to select virtual edges
        
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
            if params is not None:
                raise ValueError('Setting parameters not yet supported for mayer_layers')

            drop_layer = lambda p: MayrDropout(p)
            linear_layer = lambda input_dim, output_dim, p: MayrLinear(input_dim, output_dim, p)
        else:
            drop_layer = lambda p: nn.Dropout(p)

            def linear_layer(input_dim: int, output_dim: int, p: float, idx: int):
                if params is not None:
                    return MAMLLinear(
                                   weight=params['ffn.{}.weight'.format(idx)],
                                   bias=params['ffn.{}.bias'.format(idx)])
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

        if args.dataset_type == 'kernel':
            self.kernel_output_layer = LearnedKernel(args)
        
        if args.gradual_unfreezing:
            self.create_unfreeze_queue(args)

    def create_unfreeze_queue(self, args, freeze=True):
        if hasattr(self, 'encoder') and hasattr(self, 'ffn'):  # do this once encoder and ffn both initialized
            if args.diff_depth_weights:
                self.unfreeze_queue = []
                for depth in range(len(self.encoder.encoder.W_h[0])):
                    for layer_per_depth in range(len(self.encoder.encoder.W_h)):
                        self.unfreeze_queue.append(self.encoder.encoder.W_h[layer_per_depth][depth])
            else:
                self.unfreeze_queue = [self.encoder]
            for ffn_component in self.ffn:
                if isinstance(ffn_component, nn.Linear):
                    self.unfreeze_queue.append(ffn_component)
            if freeze:
                for param_group in self.unfreeze_queue:
                    for param in param_group.parameters():
                        param.requires_grad = False
    
    def unfreeze_next(self):
        if len(self.unfreeze_queue) == 0:
            return False
        layer_to_unfreeze = self.unfreeze_queue.pop(-1)
        for param in layer_to_unfreeze.parameters():
            param.requires_grad = True
        return True

    def forward(self, *input):
        return self.ffn(self.encoder(*input))


def build_model(args: Namespace, params: Dict[str, nn.Parameter] = None) -> nn.Module:
    """
    Builds a message passing neural network including final linear layers and initializes parameters.

    :param args: Arguments.
    :param params: Parameters to use instead of creating parameters.
    :return: An nn.Module containing the MPN encoder along with final linear layers with parameters initialized.
    """
    # Regression with binning
    if args.dataset_type == 'regression_with_binning':
        output_size = args.num_bins * args.num_tasks
    elif args.dataset_type == 'unsupervised':
        output_size = args.unsupervised_n_clusters
    elif args.dataset_type == 'kernel':
        output_size = args.ffn_hidden_size  # there will be another output layer later, for the pair of encodings
    else:
        output_size = args.num_tasks
    if args.maml:
        output_size = 1
    args.output_size = output_size

    if args.moe:
        if params is not None:
            raise ValueError('Setting parameters not yet supported for MOE or GAN')

        model = MOE(args)
        if args.adversarial:
            model = GAN(args, prediction_model=model, encoder=model.encoder)
        initialize_weights(model)

        return model

    model = MoleculeModel()
    model.create_encoder(args, params=params)
    model.create_ffn(args, params=params)

    if args.adversarial:
        if params is not None:
            raise ValueError('Setting parameters not yet supported for GAN')

        model = GAN(args, prediction_model=model, encoder=model.encoder)

    initialize_weights(model)

    return model
