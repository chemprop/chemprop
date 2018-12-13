from argparse import Namespace

import torch.nn as nn

from .gan import GAN
from .jtnn import JTNN
from .moe import MOE
from .mpn import MPN
from .learned_kernel import LearnedKernel
from chemprop.nn_utils import get_activation_function, initialize_weights, MayrDropout, MayrLinear


class MoleculeModel(nn.Module):
    def __init__(self):
        super(MoleculeModel, self).__init__()
    
    def create_encoder(self, args: Namespace):
        if args.jtnn:
            self.encoder = JTNN(args)
        elif args.dataset_type == 'bert_pretraining':
            self.encoder = MPN(args, graph_input=True)
        else:
            self.encoder = MPN(args)
        
        if args.freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
        
        if args.gradual_unfreezing:
            self.create_unfreeze_queue(args)

    def create_ffn(self, args: Namespace):
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
            drop_layer = lambda p: MayrDropout(p)
            linear_layer = lambda input_dim, output_dim, p: MayrLinear(input_dim, output_dim, p)
        else:
            drop_layer = lambda p: nn.Dropout(p)
            linear_layer = lambda input_dim, output_dim, p: nn.Linear(input_dim, output_dim)

        # Create FFN layers
        if args.ffn_num_layers == 1:
            ffn = [
                drop_layer(args.ffn_input_dropout),
                linear_layer(first_linear_dim, args.output_size, args.ffn_input_dropout)
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
                linear_layer(args.ffn_hidden_size, args.output_size, args.ffn_dropout),
            ])

        # Classification
        if args.dataset_type == 'classification':
            ffn.append(nn.Sigmoid())

        # Combined model
        ffn = nn.Sequential(*ffn)

        self.ffn = ffn

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


def build_model(args: Namespace) -> nn.Module:
    """
    Builds a message passing neural network including final linear layers and initializes parameters.

    :param args: Arguments.
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
    args.output_size = output_size

    if args.moe:
        model = MOE(args)
        if args.adversarial:
            model = GAN(args, prediction_model=model, encoder=model.encoder)
        initialize_weights(model)

        return model

    model = MoleculeModel()
    model.create_encoder(args)
    model.create_ffn(args)

    if args.adversarial:
        model = GAN(args, prediction_model=model, encoder=model.encoder)

    initialize_weights(model)

    return model
