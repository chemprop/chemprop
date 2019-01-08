from argparse import Namespace
from collections import defaultdict
from functools import partial
from typing import Dict, List, Union

import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np

from chemprop.features import BatchMolGraph, get_atom_fdim, get_bond_fdim, mol2graph
from chemprop.nn_utils import create_mask, index_select_ND, visualize_atom_attention, visualize_bond_attention, \
    get_activation_function


class MPNEncoder(nn.Module):
    """A message passing neural network for encoding a molecule."""

    def __init__(self,
                 args: Namespace,
                 atom_fdim: int,
                 bond_fdim: int,
                 params: Dict[str, nn.Parameter] = None,
                 param_prefix: str = 'encoder.encoder.'):
        """Initializes the MPN.

        :param args: Arguments.
        :param atom_fdim: Atom feature dimension.
        :param bond_fdim: Bond feature dimension.
        :param params: Parameters to use instead of creating parameters.
        :param param_prefix: Prefix of parameter names.
        """
        super(MPNEncoder, self).__init__()
        self.atom_fdim = atom_fdim
        self.bond_fdim = bond_fdim
        self.param_prefix = param_prefix
        self.hidden_size = args.hidden_size
        self.bias = args.bias
        self.depth = args.depth
        self.diff_depth_weights = args.diff_depth_weights
        self.layers_per_message = args.layers_per_message
        self.normalize_messages = args.normalize_messages
        self.use_layer_norm = args.layer_norm
        self.dropout = args.dropout
        self.attention = args.attention
        self.message_attention = args.message_attention
        self.global_attention = args.global_attention
        self.message_attention_heads = args.message_attention_heads
        self.master_node = args.master_node
        self.master_dim = args.master_dim
        self.use_master_as_output = args.use_master_as_output
        self.deepset = args.deepset
        self.set2set = args.set2set
        self.set2set_iters = args.set2set_iters
        self.learn_virtual_edges = args.learn_virtual_edges
        self.bert_pretraining = args.dataset_type == 'bert_pretraining'
        if self.bert_pretraining:
            self.output_size = args.vocab.output_size
            self.features_size = args.features_size
        self.undirected = args.undirected
        self.atom_messages = args.atom_messages
        self.features_only = args.features_only
        self.use_input_features = args.use_input_features
        self.args = args

        if self.features_only:
            return  # won't use any of the graph stuff in this case

        if self.atom_messages:
            # Not supported with atom messages
            assert not (self.message_attention or self.global_attention or self.learn_virtual_edges or
                        self.master_node or self.bert_pretraining or self.undirected)
            assert self.layers_per_message == 1

        # Layer norm
        if self.use_layer_norm:
            self.layer_norm = nn.LayerNorm(self.hidden_size)

        # Dropout
        self.dropout_layer = nn.Dropout(p=self.dropout)

        # Activation
        self.act_func = get_activation_function(args.activation)

        # Using pre-specified parameters (for meta learning)
        if params is not None:
            params = defaultdict(lambda: None, params)  # nonexistent parameters default to None

            self.cached_zero_vector = params[self.param_prefix + 'cached_zero_vector']
            self.W_i = partial(F.linear,
                               weight=params[self.param_prefix + 'W_i.weight'],
                               bias=params[self.param_prefix + 'W_i.bias'])
            if self.message_attention:
                self.num_heads = self.message_attention_heads
                self.W_ma = [partial(F.linear,
                                    weight=params[self.param_prefix + 'W_ma.{}.weight'.format(i)],
                                    bias=params[self.param_prefix + 'W_ma.{}.bias'.format(i)])
                            for i in range(self.num_heads)]
            if args.learn_virtual_edges:
                self.lve = partial(F.linear,
                                weight=params[self.param_prefix + 'lve.weight'],
                                bias=params[self.param_prefix + 'lve.bias'])
            if self.diff_depth_weights:
                self.W_h = [[partial(F.linear,
                                     weight=params[self.param_prefix + 'W_h.{}.{}.weight'.format(i, j)],
                                     bias=params[self.param_prefix + 'W_h.{}.{}.bias'.format(i, j)])
                             for j in range(self.depth - 1)]
                            for i in range(self.layers_per_message)]
            else:
                # TODO this option is currently broken; the params are None
                self.W_h = [[partial(F.linear,
                                     weight=params[self.param_prefix + 'W_h.{}.{}.weight'.format(i, j)],
                                     bias=params[self.param_prefix + 'W_h.{}.{}.bias'.format(i, j)])
                             for j in range(self.depth - 1)]
                            for i in range(self.layers_per_message)]
            self.W_ga1 = partial(F.linear,
                                 weight=params[self.param_prefix + 'W_ga1.weight'],
                                 bias=params[self.param_prefix + 'W_ga1.bias'])
            self.W_ga2 = partial(F.linear,
                                 weight=params[self.param_prefix + 'W_ga2.weight'],
                                 bias=params[self.param_prefix + 'W_ga2.bias'])
            self.W_master_in = partial(F.linear,
                                       weight=params[self.param_prefix + 'W_master_in.weight'],
                                       bias=params[self.param_prefix + 'W_master_in.bias'])
            self.W_master_out = partial(F.linear,
                                        weight=params[self.param_prefix + 'W_master_out.weight'],
                                        bias=params[self.param_prefix + 'W_master_out.bias'])
            self.W_o = partial(F.linear,
                               weight=params[self.param_prefix + 'W_o.weight'],
                               bias=params[self.param_prefix + 'W_o.bias'])
            self.W_s2s_a = partial(F.linear,
                                   weight=params[self.param_prefix + 'W_s2s_a.weight'],
                                   bias=params[self.param_prefix + 'W_s2s_a.bias'])
            self.W_s2s_b = partial(F.linear,
                                   weight=params[self.param_prefix + 'W_s2s_b.weight'],
                                   bias=params[self.param_prefix + 'W_s2s_b.bias'])
            if self.set2set:
                raise ValueError('Setting params of LSTM not supported yet.')
            self.W_a = partial(F.linear,
                               weight=params[self.param_prefix + 'W_a.weight'],
                               bias=params[self.param_prefix + 'W_a.bias'])
            self.W_b = partial(F.linear,
                               weight=params[self.param_prefix + 'W_b.weight'],
                               bias=params[self.param_prefix + 'W_b.bias'])
            self.W_v = partial(F.linear,
                               weight=params[self.param_prefix + 'W_v.weight'],
                               bias=params[self.param_prefix + 'W_v.bias'])
            self.W_f = partial(F.linear,
                               weight=params[self.param_prefix + 'W_f.weight'],
                               bias=params[self.param_prefix + 'W_f.bias'])

            return

        # Cached zeros
        self.cached_zero_vector = nn.Parameter(torch.zeros(self.hidden_size), requires_grad=False)

        # Input
        input_dim = self.atom_fdim if self.atom_messages else self.bond_fdim
        self.W_i = nn.Linear(input_dim, self.hidden_size, bias=self.bias)

        # Message attention
        if self.message_attention:
            self.num_heads = self.message_attention_heads
            w_h_input_size = self.num_heads * self.hidden_size
            self.W_ma = nn.ModuleList([nn.Linear(self.hidden_size, self.hidden_size, bias=self.bias)
                                       for _ in range(self.num_heads)])
        elif self.atom_messages:
            w_h_input_size = self.hidden_size + self.bond_fdim
        else:
            w_h_input_size = self.hidden_size

        if self.learn_virtual_edges:
            self.lve = nn.Linear(self.atom_fdim, self.atom_fdim)

        # Message passing
        if self.diff_depth_weights:
            # Different weight matrix for each depth
            self.W_h = nn.ModuleList([
                nn.ModuleList([
                    nn.Linear(w_h_input_size, self.hidden_size, bias=self.bias) for _ in range(self.depth - 1)
                ]) for _ in range(self.layers_per_message)
            ])
        else:
            # Shared weight matrix across depths (default)
            self.W_h = nn.ModuleList([nn.ModuleList([
                nn.Linear(w_h_input_size, self.hidden_size, bias=self.bias)] * (self.depth - 1))
                for _ in range(self.layers_per_message)
            ])

        if self.global_attention:
            self.W_ga1 = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
            self.W_ga2 = nn.Linear(self.hidden_size, self.hidden_size)

        if self.master_node:
            self.W_master_in = nn.Linear(self.hidden_size, self.master_dim)
            self.W_master_out = nn.Linear(self.master_dim, self.hidden_size)

        # Readout
        if not (self.master_node and self.use_master_as_output):
            self.W_o = nn.Linear(self.atom_fdim + self.hidden_size, self.hidden_size)

        if self.deepset:
            self.W_s2s_a = nn.Linear(self.hidden_size, self.hidden_size, bias=self.bias)
            self.W_s2s_b = nn.Linear(self.hidden_size, self.hidden_size, bias=self.bias)

        if self.set2set:
            self.set2set_rnn = nn.LSTM(
                input_size=self.hidden_size,
                hidden_size=self.hidden_size,
                dropout=self.dropout,
                bias=False  # no bias so that an input of all zeros stays all zero
            )

        if self.attention:
            self.W_a = nn.Linear(self.hidden_size, self.hidden_size, bias=self.bias)
            self.W_b = nn.Linear(self.hidden_size, self.hidden_size)

        if self.bert_pretraining:
            if args.bert_vocab_func == 'feature_vector':
                self.W_v = nn.Linear(self.hidden_size, args.vocab.output_size)
            else:
                self.W_v = nn.Linear(self.hidden_size, self.output_size)

            if self.features_size is not None:
                self.W_f = nn.Linear(self.hidden_size, self.features_size)

    def forward(self,
                mol_graph: BatchMolGraph,
                features_batch: List[np.ndarray] = None,
                viz_dir: str = None) -> Union[torch.FloatTensor, Dict[str, torch.FloatTensor]]:
        """
        Encodes a batch of molecular graphs.

        :param mol_graph: A BatchMolGraph representing a batch of molecular graphs.
        :param features_batch: A list of ndarrays containing additional features.
        :param viz_dir: Directory in which to save visualized attention weights.
        :return: A PyTorch tensor of shape (num_molecules, hidden_size) containing the encoding of each molecule.
        """
        if self.use_input_features:
            features_batch = torch.from_numpy(np.stack(features_batch)).float()

            if self.args.cuda:
                features_batch = features_batch.cuda()

            if self.features_only:
                return features_batch

        f_atoms, f_bonds, a2b, b2a, b2revb, a_scope, b_scope = mol_graph.get_components()

        if self.atom_messages:
            a2a = mol_graph.get_a2a()

        if self.args.cuda or next(self.parameters()).is_cuda:
            f_atoms, f_bonds, a2b, b2a, b2revb = f_atoms.cuda(), f_bonds.cuda(), a2b.cuda(), b2a.cuda(), b2revb.cuda()

            if self.atom_messages:
                a2a = a2a.cuda()

        if self.learn_virtual_edges:
            atom1_features, atom2_features = f_atoms[b2a], f_atoms[b2a[b2revb]]  # each num_bonds x atom_fdim
            ve_score = torch.sum(self.lve(atom1_features) * atom2_features, dim=1) + torch.sum(self.lve(atom2_features) * atom1_features, dim=1)
            is_ve_indicator_index = self.atom_fdim  # in current featurization, the first bond feature is 1 or 0 for virtual or not virtual
            num_virtual = f_bonds[:, is_ve_indicator_index].sum()
            straight_through_mask = torch.ones(f_bonds.size(0)).cuda() + f_bonds[:, is_ve_indicator_index] * (ve_score - ve_score.detach()) / num_virtual  # normalize for grad norm
            straight_through_mask = straight_through_mask.unsqueeze(1).repeat((1, self.hidden_size))  # num_bonds x hidden_size

        # Input
        if self.atom_messages:
            input = self.W_i(f_atoms)  # num_atoms x hidden_size
        else:
            input = self.W_i(f_bonds)  # num_bonds x hidden_size
        message = self.act_func(input)  # num_bonds x hidden_size

        if self.message_attention:
            b2b = mol_graph.get_b2b()  # Warning: this is O(n_atoms^3) when using virtual edges

            if next(self.parameters()).is_cuda:
                b2b = b2b.cuda()

            message_attention_mask = (b2b != 0).float()  # num_bonds x max_num_bonds

        if self.global_attention:
            global_attention_mask = torch.zeros(mol_graph.n_bonds, mol_graph.n_bonds)  # num_bonds x num_bonds

            for start, length in b_scope:
                for i in range(start, start + length):
                    global_attention_mask[i, start:start + length] = 1

            if next(self.parameters()).is_cuda:
                global_attention_mask = global_attention_mask.cuda()

        # Message passing
        for depth in range(self.depth - 1):
            if self.undirected:
                message = (message + message[b2revb]) / 2

            if self.learn_virtual_edges:
                message = message * straight_through_mask

            if self.message_attention:
                # TODO: Parallelize attention heads
                nei_message = index_select_ND(message, b2b)
                message = message.unsqueeze(1).repeat((1, nei_message.size(1), 1))  # num_bonds x maxnb x hidden
                attention_scores = [(self.W_ma[i](nei_message) * message).sum(dim=2)
                                    for i in range(self.num_heads)]  # num_bonds x maxnb
                attention_scores = [attention_scores[i] * message_attention_mask + (1 - message_attention_mask) * (-1e+20)
                                    for i in range(self.num_heads)]  # num_bonds x maxnb
                attention_weights = [F.softmax(attention_scores[i], dim=1)
                                     for i in range(self.num_heads)]  # num_bonds x maxnb
                message_components = [nei_message * attention_weights[i].unsqueeze(2).repeat((1, 1, self.hidden_size))
                                      for i in range(self.num_heads)]  # num_bonds x maxnb x hidden
                message_components = [component.sum(dim=1) for component in message_components]  # num_bonds x hidden
                message = torch.cat(message_components, dim=1)  # num_bonds x num_heads * hidden
            elif self.atom_messages:
                nei_a_message = index_select_ND(message, a2a)  # num_atoms x max_num_bonds x hidden
                nei_f_bonds = index_select_ND(f_bonds, a2b)  # num_atoms x max_num_bonds x bond_fdim
                nei_message = torch.cat((nei_a_message, nei_f_bonds), dim=2)  # num_atoms x max_num_bonds x hidden + bond_fdim
                message = nei_message.sum(dim=1)  # num_atoms x hidden + bond_fdim
            else:
                # m(a1 -> a2) = [sum_{a0 \in nei(a1)} m(a0 -> a1)] - m(a2 -> a1)
                # message      a_message = sum(nei_a_message)      rev_message
                nei_a_message = index_select_ND(message, a2b)  # num_atoms x max_num_bonds x hidden
                a_message = nei_a_message.sum(dim=1)  # num_atoms x hidden
                rev_message = message[b2revb]  # num_bonds x hidden
                message = a_message[b2a] - rev_message  # num_bonds x hidden

            for lpm in range(self.layers_per_message - 1):
                message = self.W_h[lpm][depth](message)  # num_bonds x hidden
                message = self.act_func(message)
            message = self.W_h[self.layers_per_message - 1][depth](message)

            if self.normalize_messages:
                message = message / message.norm(dim=1, keepdim=True)

            if self.master_node:
                # master_state = self.W_master_in(self.act_func(nei_message.sum(dim=0))) #try something like this to preserve invariance for master node
                # master_state = self.GRU_master(nei_message.unsqueeze(1))
                # master_state = master_state[-1].squeeze(0) #this actually doesn't preserve order invariance anymore
                mol_vecs = [self.cached_zero_vector]
                for start, size in b_scope:
                    if size == 0:
                        continue
                    mol_vec = message.narrow(0, start, size)
                    mol_vec = mol_vec.sum(dim=0) / size
                    mol_vecs += [mol_vec for _ in range(size)]
                master_state = self.act_func(self.W_master_in(torch.stack(mol_vecs, dim=0)))  # num_bonds x hidden_size
                message = self.act_func(input + message + self.W_master_out(master_state))  # num_bonds x hidden_size
            else:
                message = self.act_func(input + message)  # num_bonds x hidden_size

            if self.global_attention:
                attention_scores = torch.matmul(self.W_ga1(message), message.t())  # num_bonds x num_bonds
                attention_scores = attention_scores * global_attention_mask + (1 - global_attention_mask) * (-1e+20)  # num_bonds x num_bonds
                attention_weights = F.softmax(attention_scores, dim=1)  # num_bonds x num_bonds
                attention_hiddens = torch.matmul(attention_weights, message)  # num_bonds x hidden_size
                attention_hiddens = self.act_func(self.W_ga2(attention_hiddens))  # num_bonds x hidden_size
                attention_hiddens = self.dropout_layer(attention_hiddens)  # num_bonds x hidden_size
                message = message + attention_hiddens  # num_bonds x hidden_size

                if viz_dir is not None:
                    visualize_bond_attention(viz_dir, mol_graph, attention_weights, depth)

            if self.use_layer_norm:
                message = self.layer_norm(message)

            message = self.dropout_layer(message)  # num_bonds x hidden

        if self.master_node and self.use_master_as_output:
            assert self.hidden_size == self.master_dim
            mol_vecs = []
            for start, size in b_scope:
                if size == 0:
                    mol_vecs.append(self.cached_zero_vector)
                else:
                    mol_vecs.append(master_state[start])
            return torch.stack(mol_vecs, dim=0)

        # Get atom hidden states from message hidden states
        if self.learn_virtual_edges:
            message = message * straight_through_mask

        a2x = a2a if self.atom_messages else a2b
        nei_a_message = index_select_ND(message, a2x)  # num_atoms x max_num_bonds x hidden
        a_message = nei_a_message.sum(dim=1)  # num_atoms x hidden
        a_input = torch.cat([f_atoms, a_message], dim=1)  # num_atoms x (atom_fdim + hidden)
        atom_hiddens = self.act_func(self.W_o(a_input))  # num_atoms x hidden
        atom_hiddens = self.dropout_layer(atom_hiddens)  # num_atoms x hidden

        if self.deepset:
            atom_hiddens = self.W_s2s_a(atom_hiddens)
            atom_hiddens = self.act_func(atom_hiddens)
            atom_hiddens = self.W_s2s_b(atom_hiddens)

        if self.bert_pretraining:
            atom_preds = self.W_v(atom_hiddens)[1:]  # num_atoms x vocab/output size (leave out atom padding)

        # Readout
        if self.set2set:
            # Set up sizes
            batch_size = len(a_scope)
            lengths = [length for _, length in a_scope]
            max_num_atoms = max(lengths)

            # Set up memory from atom features
            memory = torch.zeros(batch_size, max_num_atoms, self.hidden_size)  # (batch_size, max_num_atoms, hidden_size)
            for i, (start, size) in enumerate(a_scope):
                memory[i, :size] = atom_hiddens.narrow(0, start, size)
            memory_transposed = memory.transpose(2, 1)  # (batch_size, hidden_size, max_num_atoms)

            # Create mask (1s for atoms, 0s for not atoms)
            mask = create_mask(lengths, cuda=next(self.parameters()).is_cuda)  # (max_num_atoms, batch_size)
            mask = mask.t().unsqueeze(2)  # (batch_size, max_num_atoms, 1)

            # Set up query
            query = torch.ones(1, batch_size, self.hidden_size)  # (1, batch_size, hidden_size)

            # Move to cuda
            if next(self.parameters()).is_cuda:
                memory, memory_transposed, query = memory.cuda(), memory_transposed.cuda(), query.cuda()

            # Run RNN
            for _ in range(self.set2set_iters):
                # Compute attention weights over atoms in each molecule
                query = query.squeeze(0).unsqueeze(2)  # (batch_size,  hidden_size, 1)
                dot = torch.bmm(memory, query)  # (batch_size, max_num_atoms, 1)
                dot = dot * mask + (1 - mask) * (-1e+20)  # (batch_size, max_num_atoms, 1)
                attention = F.softmax(dot, dim=1)  # (batch_size, max_num_atoms, 1)

                # Construct next input as attention over memory
                attended = torch.bmm(memory_transposed, attention)  # (batch_size, hidden_size, 1)
                attended = attended.view(1, batch_size, self.hidden_size)  # (1, batch_size, hidden_size)

                # Run RNN for one step
                query, _ = self.set2set_rnn(attended)  # (1, batch_size, hidden_size)

            # Final RNN output is the molecule encodings
            mol_vecs = query.squeeze(0)  # (batch_size, hidden_size)
        else:
            mol_vecs = []
            # TODO: Maybe do this in parallel with masking rather than looping
            for i, (a_start, a_size) in enumerate(a_scope):
                if a_size == 0:
                    mol_vecs.append(self.cached_zero_vector)
                else:
                    cur_hiddens = atom_hiddens.narrow(0, a_start, a_size)

                    if self.attention:
                        att_w = torch.matmul(self.W_a(cur_hiddens), cur_hiddens.t())
                        att_w = F.softmax(att_w, dim=1)
                        att_hiddens = torch.matmul(att_w, cur_hiddens)
                        att_hiddens = self.act_func(self.W_b(att_hiddens))
                        att_hiddens = self.dropout_layer(att_hiddens)
                        mol_vec = (cur_hiddens + att_hiddens)

                        if viz_dir is not None:
                            visualize_atom_attention(viz_dir, mol_graph.smiles_batch[i], a_size, att_w)
                    else:
                        mol_vec = cur_hiddens  # (num_atoms, hidden_size)

                    mol_vec = mol_vec.sum(dim=0) / a_size
                    mol_vecs.append(mol_vec)

            mol_vecs = torch.stack(mol_vecs, dim=0)  # (num_molecules, hidden_size)
        
        if self.use_input_features:
            features_batch = features_batch.to(mol_vecs)
            if len(features_batch.shape) == 1:
                features_batch = features_batch.view([1,features_batch.shape[0]])
            mol_vecs = torch.cat([mol_vecs, features_batch], dim=1)  # (num_molecules, hidden_size)

        if self.bert_pretraining:
            features_preds = self.W_f(mol_vecs) if hasattr(self, 'W_f') else None
            return {
                'features': features_preds,
                'vocab': atom_preds
            }

        return mol_vecs  # num_molecules x hidden


class MPN(nn.Module):
    """A message passing neural network for encoding a molecule."""

    def __init__(self,
                 args: Namespace,
                 atom_fdim: int = None,
                 bond_fdim: int = None,
                 graph_input: bool = False,
                 params: Dict[str, nn.Parameter] = None):
        super(MPN, self).__init__()
        self.args = args
        self.atom_fdim = atom_fdim or get_atom_fdim(args)
        self.bond_fdim = bond_fdim or get_bond_fdim(args) + (not args.atom_messages) * self.atom_fdim
        self.graph_input = graph_input
        self.encoder = MPNEncoder(self.args, self.atom_fdim, self.bond_fdim, params=params)

    def forward(self,
                batch: Union[List[str], BatchMolGraph],
                features_batch: List[np.ndarray] = None) -> torch.Tensor:
        """
        Encodes a batch of molecular SMILES strings.

        :param batch: A list of SMILES strings or a BatchMolGraph (if self.graph_input).
        :param features_batch: A list of ndarrays containing additional features.
        :return: A PyTorch tensor of shape (num_molecules, hidden_size) containing the encoding of each molecule.
        """
        if not self.graph_input and not self.arg.features_only:  # if features only, batch won't even be used
            batch = mol2graph(batch, self.args)

        output = self.encoder.forward(batch, features_batch)

        if self.args.adversarial:
            self.saved_encoder_output = output

        return output

    def viz_attention(self,
                      viz_dir: str,
                      batch: Union[List[str], BatchMolGraph],
                      features_batch: List[np.ndarray] = None):
        """
        Visualizes attention weights for a batch of molecular SMILES strings

        :param viz_dir: Directory in which to save visualized attention weights.
        :param batch: A list of SMILES strings or a BatchMolGraph (if self.graph_input).
        :param features_batch: A list of ndarrays containing additional features.
        """
        if not self.graph_input:
            batch = mol2graph(batch, self.args)

        self.encoder.forward(batch, features_batch, viz_dir=viz_dir)
