from argparse import Namespace
from typing import List, Tuple

import torch
import torch.nn.functional as F
import torch.nn as nn

from featurization import get_atom_fdim, get_bond_fdim
from nn_utils import create_mask, index_select_ND


class MPN(nn.Module):
    """A message passing neural network for encoding a molecule."""

    def __init__(self, args: Namespace):
        """Initializes the MPN."""
        super(MPN, self).__init__()
        self.hidden_size = args.hidden_size
        self.bias = args.bias
        self.depth = args.depth
        self.dropout = args.dropout
        self.attention = args.attention
        self.message_attention = args.message_attention
        self.message_attention_heads = args.message_attention_heads
        self.master_node = args.master_node
        self.master_dim = args.master_dim
        self.deepset = args.deepset
        self.set2set = args.set2set
        self.set2set_iters = args.set2set_iters

        # Input
        self.W_i = nn.Linear(get_atom_fdim(args) + get_bond_fdim(args), self.hidden_size, bias=self.bias)

        # Message passing
        if self.message_attention:
            self.num_heads = self.message_attention_heads
            self.W_h = nn.Linear(self.num_heads * self.hidden_size, self.hidden_size, bias=self.bias)
            self.W_ma = nn.ModuleList([nn.Linear(self.hidden_size, self.hidden_size, bias=self.bias)
                                       for _ in range(self.num_heads)])
            # uncomment this later if you want attention over binput + nei_message? or on atom incoming at end
            # self.W_ma2 = nn.Linear(hidden_size, 1, bias=self.bias)
        else:
            self.W_h = nn.Linear(self.hidden_size, self.hidden_size, bias=self.bias)

        if self.master_node:
            # self.GRU_master = nn.GRU(self.hidden_size, self.master_dim)
            self.W_master_in = nn.Linear(self.hidden_size, self.master_dim)
            self.W_master_out = nn.Linear(self.master_dim, self.hidden_size)
            # self.layer_norm = nn.LayerNorm(self.hidden_size)

        # Readout
        self.W_o = nn.Linear(get_atom_fdim(args) + self.hidden_size, self.hidden_size)

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

        # Dropout
        self.dropout_layer = nn.Dropout(p=self.dropout)

        # Activation
        if args.activation == 'ReLU':
            self.act_func = nn.ReLU()
        elif args.activation == 'LeakyReLU':
            self.act_func = nn.LeakyReLU(0.1)
        elif args.activation == 'PReLU':
            self.act_func = nn.PReLU()
        elif args.activation == 'tanh':
            self.act_func = nn.Tanh()
        else:
            raise ValueError('Activation "{}" not supported.'.format(args.activation))

    def forward(self, mol_graph: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, List[Tuple[int, int]], List[Tuple[int, int]]]) -> torch.Tensor:
        """
        Encodes a batch of molecular graphs.

        :param mol_graph: A tuple containing a batch of molecular graphs (see mol2graph docstring for details).
        :return: A PyTorch tensor of shape (num_molecules, hidden_size) containing the encoding of each molecule.
        """
        fatoms, fbonds, agraph, bgraph, ascope, bscope = mol_graph
        if next(self.parameters()).is_cuda:
            fatoms, fbonds, agraph, bgraph = fatoms.cuda(), fbonds.cuda(), agraph.cuda(), bgraph.cuda()

        # Input
        binput = self.W_i(fbonds)
        message = self.act_func(binput)

        # Message passing
        for i in range(self.depth - 1):
            nei_message = index_select_ND(message, bgraph)
            if self.message_attention:
                message = message.unsqueeze(1).repeat((1, nei_message.size(1), 1))  # num_bonds x 1 x hidden
                attention_scores = [(self.W_ma[i](nei_message) * message).sum(dim=2)
                                    for i in range(self.num_heads)]  # num_bonds x maxnb
                attention_weights = [F.softmax(attention_scores[i], dim=1)
                                     for i in range(self.num_heads)]  # num_bonds x maxnb
                message_components = [nei_message * attention_weights[i].unsqueeze(2).repeat((1, 1, self.hidden_size))
                                      for i in range(self.num_heads)]  # num_bonds x maxnb x hidden
                message_components = [component.sum(dim=1) for component in message_components]  # num_bonds x hidden
                nei_message = torch.cat(message_components, dim=1)  # num_bonds x 3*hidden
            else:
                nei_message = nei_message.sum(dim=1)  # num_bonds x hidden
            nei_message = self.W_h(nei_message)
            if self.master_node:
                # master_state = self.W_master_in(self.act_func(nei_message.sum(dim=0))) #try something like this to preserve invariance for master node
                # master_state = self.GRU_master(nei_message.unsqueeze(1))
                # master_state = master_state[-1].squeeze(0) #this actually doesn't preserve order invariance anymore
                mol_vecs = [torch.zeros(self.hidden_size).cuda()]
                for start, size in bscope:
                    mol_vec = nei_message.narrow(0, start, size)
                    mol_vec = mol_vec.sum(dim=0) / size
                    mol_vecs += [mol_vec for _ in range(size)]
                master_state = self.act_func(self.W_master_in(torch.stack(mol_vecs, dim=0)))  # (num_bonds, hidden_size)
                message = self.act_func(binput + nei_message + self.W_master_out(master_state))
                # message = self.layer_norm(message)
            else:
                message = self.act_func(binput + nei_message)
            message = self.dropout_layer(message)  # num_bonds x hidden

        # Get atom hidden states from message hidden states
        nei_message = index_select_ND(message, agraph)
        nei_message = nei_message.sum(dim=1)
        ainput = torch.cat([fatoms, nei_message], dim=1)
        atom_hiddens = self.act_func(self.W_o(ainput))
        atom_hiddens = self.dropout_layer(atom_hiddens)

        # Readout
        if self.set2set:
            # Set up sizes
            batch_size = len(ascope)
            lengths = [length for _, length in ascope]
            max_num_atoms = max(lengths)

            # Set up memory from atom features
            memory = torch.zeros(batch_size, max_num_atoms, self.hidden_size)  # (batch_size, max_num_atoms, hidden_size)
            for i, (start, size) in enumerate(ascope):
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
            for start, size in ascope:
                cur_hiddens = atom_hiddens.narrow(0, start, size)

                if self.attention:
                    att_w = torch.matmul(self.W_a(cur_hiddens), cur_hiddens.t())
                    att_w = F.softmax(att_w, dim=1)
                    att_hiddens = torch.matmul(att_w, cur_hiddens)
                    att_hiddens = self.act_func(self.W_b(att_hiddens))
                    att_hiddens = self.dropout_layer(att_hiddens)
                    mol_vec = (cur_hiddens + att_hiddens)
                else:
                    mol_vec = cur_hiddens  # (num_atoms, hidden_size)

                if self.deepset:
                    mol_vec = self.W_s2s_a(mol_vec)
                    mol_vec = self.act_func(mol_vec)
                    mol_vec = self.W_s2s_b(mol_vec)

                mol_vec = mol_vec.sum(dim=0) / size
                mol_vecs.append(mol_vec)

            mol_vecs = torch.stack(mol_vecs, dim=0)  # (num_molecules, hidden_size)

        return mol_vecs  # (num_molecules, hidden_size)
