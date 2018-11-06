from argparse import Namespace
from typing import List

import torch
import torch.nn as nn
import torch.autograd as autograd
from torch.optim import Adam

from chemprop.nn_utils import NoamLR


class GAN(nn.Module):
    def __init__(self, args: Namespace, prediction_model: nn.Module, encoder: nn.Module):
        super(GAN, self).__init__()
        self.args = args
        self.prediction_model = prediction_model
        self.encoder = encoder

        self.hidden_size = args.hidden_size
        self.disc_input_size = args.hidden_size + args.output_size
        self.act_func = self.encoder.encoder.act_func

        self.netD = nn.Sequential(
            nn.Linear(self.disc_input_size, self.hidden_size), # doesn't support jtnn or additional features rn
            self.act_func,
            nn.Linear(self.hidden_size, self.hidden_size),
            self.act_func,
            nn.Linear(self.hidden_size, self.hidden_size),
            self.act_func,
            nn.Linear(self.hidden_size, 1)
        )
        self.beta = args.wgan_beta

        # the optimizers don't really belong here, but we put it here so that we don't clutter code for other opts
        self.optimizerG = Adam(self.encoder.parameters(), lr=args.init_lr * args.gan_lr_mult, betas=(0, 0.9))
        self.optimizerD = Adam(self.netD.parameters(), lr=args.init_lr * args.gan_lr_mult, betas=(0, 0.9))

        self.use_scheduler = args.gan_use_scheduler
        if self.use_scheduler:
            self.schedulerG = NoamLR(
                self.optimizerG,
                warmup_epochs=args.warmup_epochs,
                total_epochs=args.epochs,
                steps_per_epoch=args.train_data_length // args.batch_size,
                init_lr=args.init_lr * args.gan_lr_mult,
                max_lr=args.max_lr * args.gan_lr_mult,
                final_lr=args.final_lr * args.gan_lr_mult
            )
            self.schedulerD = NoamLR(
                self.optimizerD,
                warmup_epochs=args.warmup_epochs,
                total_epochs=args.epochs,
                steps_per_epoch=(args.train_data_length // args.batch_size) * args.gan_d_per_g,
                init_lr=args.init_lr * args.gan_lr_mult,
                max_lr=args.max_lr * args.gan_lr_mult,
                final_lr=args.final_lr * args.gan_lr_mult
            )
        
    def forward(self, smiles_batch: List[str], features=None) -> torch.Tensor:
        return self.prediction_model(smiles_batch, features)
    
    # TODO maybe this isn't the best way to wrap the MOE class, but it works for now
    def mahalanobis_metric(self, p, mu, j):
        return self.prediction_model.mahalanobis_metric(p, mu, j)
    
    def compute_domain_encs(self, all_train_smiles):
        return self.prediction_model.compute_domain_encs(all_train_smiles)
    
    def compute_minibatch_domain_encs(self, train_smiles):
        return self.prediction_model.compute_minibatch_domain_encs(train_smiles)

    def compute_loss(self, train_smiles, train_targets, test_smiles):
        return self.prediction_model.compute_loss(train_smiles, train_targets, test_smiles)
    
    def set_domain_encs(self, domain_encs):
        self.prediction_model.domain_encs = domain_encs
    
    def get_domain_encs(self):
        return self.prediction_model.domain_encs

    # the following methods are code borrowed from Wengong and modified
    def train_D(self, fake_smiles: List[str], real_smiles: List[str]):
        self.netD.zero_grad()

        real_output = self.prediction_model(real_smiles).detach()
        real_enc_output = self.encoder.saved_encoder_output.detach()
        real_vecs = torch.cat([real_enc_output, real_output], dim=1)
        fake_output = self.prediction_model(fake_smiles).detach()
        fake_enc_output = self.encoder.saved_encoder_output.detach()
        fake_vecs = torch.cat([fake_enc_output, fake_output], dim=1)

        # real_vecs = self.encoder(mol2graph(real_smiles, self.args)).detach()
        # fake_vecs = self.encoder(mol2graph(fake_smiles, self.args)).detach()
        real_score = self.netD(real_vecs)
        fake_score = self.netD(fake_vecs)

        score = fake_score.mean() - real_score.mean() #maximize -> minimize minus
        score.backward()

        #Gradient Penalty
        inter_gp, inter_norm = self.gradient_penalty(real_vecs, fake_vecs)
        inter_gp.backward()

        self.optimizerD.step()
        if self.use_scheduler:
            self.schedulerD.step()

        return -score.item(), inter_norm
    
    def train_G(self, fake_smiles: List[str], real_smiles: List[str]):
        self.encoder.zero_grad()

        real_output = self.prediction_model(real_smiles).detach()
        real_enc_output = self.encoder.saved_encoder_output
        real_vecs = torch.cat([real_enc_output, real_output], dim=1)
        fake_output = self.prediction_model(fake_smiles).detach()
        fake_enc_output = self.encoder.saved_encoder_output
        fake_vecs = torch.cat([fake_enc_output, fake_output], dim=1)

        # real_vecs = self.encoder(mol2graph(real_smiles, self.args))
        # fake_vecs = self.encoder(mol2graph(fake_smiles, self.args))
        real_score = self.netD(real_vecs)
        fake_score = self.netD(fake_vecs)

        score = real_score.mean() - fake_score.mean() 
        score.backward()

        self.optimizerG.step()
        if self.use_scheduler:
            self.schedulerG.step()
        self.netD.zero_grad() #technically not necessary since it'll get zero'd in the next iteration anyway

        return score.item()
    
    def gradient_penalty(self, real_vecs, fake_vecs):
        assert real_vecs.size() == fake_vecs.size()
        eps = torch.rand(real_vecs.size(0), 1).cuda()
        inter_data = eps * real_vecs + (1 - eps) * fake_vecs
        inter_data = autograd.Variable(inter_data, requires_grad=True) # TODO check if this is necessary (we detached earlier)
        inter_score = self.netD(inter_data)
        inter_score = inter_score.view(-1) # bs*hidden

        inter_grad = autograd.grad(inter_score, inter_data,
                                   grad_outputs=torch.ones(inter_score.size()).cuda(),
                                   create_graph=True,
                                   retain_graph=True,
                                   only_inputs=True)[0]

        inter_norm = inter_grad.norm(2, dim=1)
        inter_gp = ((inter_norm - 1) ** 2).mean() * self.beta

        return inter_gp, inter_norm.mean().item()
