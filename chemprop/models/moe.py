from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

from .mpn import MPN


def compute_pairwise_distances(x, y):
    """
    Computes the squared pairwise Euclidean distances between x and y.

    :param x: a tensor of shape [num_x_samples, num_features]
    :param y: a tensor of shape [num_y_samples, num_features]
    :return: a distance matrix of dimensions [num_x_samples, num_y_samples]
    Raise:
        ValueError if the inputs do not match the specified dimensions
    """
    # print('In compute_pairwise_distances  NOTE THAT THIS DOES NOT SEEM TO BE WORKING.')
    # print('x and y should have dimensions num_i_samples x num_features, it seems like the first dimension is number of atoms instead')
    # print('Uncomment print statements in utils/compute_pairwise_distances to observe this')
    # print('x.size() =', x.size())
    # print('x.size()[1] =', x.size()[1])
    # print('y.size() =', y.size())
    # print('y.size()[1] =', y.size()[1])
    # note, i copy pasted this, but this seems to be working fine for me? -yangk
    if not len(x.size()) == len(y.size()) == 2:
        raise ValueError('Both inputs should be matrices.')
    if x.size()[1] != y.size()[1]:
        raise ValueError('The number of features should be the same.')

    # By making the 'inner' dimensions of the two matrices equal to 1 using broadcasting then we are essentially
    # subtracting every pair of rows of x and y
    norm = lambda x:torch.sum(x * x, 1)
    return norm(x.unsqueeze(2) - y.t())


def gaussian_kernel_matrix(x, y, sigmas):
    """
    Computes a gaussian RBK between the samples of x and y
    We create a sum of multiple gaussian kernels each having a width sigma_i

    :param x: a tensor of shape [num_samples, num_features]
    :param y: a tensor of shape [num_samples, num_features]
    :param sigmas: a tensor of floats which denote the widths of each of the gaussians in the kernel
    :return: a tensor of shape [num_samples{x}, num_samples{y}] with the RBF kernel
    """
    beta = 1. / (2. * (sigmas.unsqueeze(1)))
    dist = compute_pairwise_distances(x, y)
    s = torch.matmul(beta, dist.view(1, -1))

    return(torch.sum(torch.exp(-s), 0)).view_as(dist)


def maximum_mean_discrepancy(x, y, kernel=gaussian_kernel_matrix):
    """
    Computes the Maximum Mean Discrepancy (MMD) of two samples: x and y
    Maximum Mean Discrepancy (MMD) is a distance-measure between the samples of the distributions of x and y. Here we
    kernel two sample estimate using the empirical mean of the two distributions

    :param x: a tensor of shape [num_samples, num_features]
    :param y: a tensor of shape [num_samples, num_features]
    :param kernel: a function which computes the kernel in MMD.  Defaults to the GaussianKernelMatrix.
    :return: a scalar denoting the squared maximum mean discrepancy loss
    """

    cost = torch.mean(kernel(x, x))
    cost += torch.mean(kernel(y, y))
    cost -= 2 * torch.mean(kernel(x, y))

    cost = torch.clamp(cost, min=0)
    return cost


class MMD(nn.Module):  # TODO(moe) can experiment with WGAN too - can try with both union of sources of with each individually
    def __init__(self, args):
        # The __init__() method is run automatically when a new instance of the MMD class is created
        # New instance created by writing new_instance = MMD(encoder, configs) -- when creating the new instance,
        # you have to pass the same arguments that are passed to the __init__() method first
        # self (the new instance) is automatically passed - self is used by convention but you can use whatever you want
        super(MMD, self).__init__()
        self.sigmas = torch.FloatTensor([
            1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1,
            1, 5, 10, 15, 20, 25, 30, 35, 100,
            1e3, 1e4, 1e5, 1e6
            ])
        if args.cuda:
            self.sigmas = self.sigmas.cuda()
        self.gaussian_kernel = partial(gaussian_kernel_matrix, sigmas=self.sigmas)

    def forward(self, hs, ht):
        loss_value = maximum_mean_discrepancy(hs, ht, kernel=self.gaussian_kernel)
        return torch.clamp(loss_value, min=1e-4)  # torch.clamp(input, min, max, out=None) --> Tensor
                                                  # This clamps all elements in input into the range [min, max] and returns a resulting tensor
                                                  # new_val = 1e-4 if old_val < 1e-4; in this case, otherwise, new_val = old_val


class HLoss(nn.Module):
    def __init__(self):
        super(HLoss, self).__init__()

    def forward(self, x):
        # b = F.softmax(x, dim=1) * F.log_softmax(x, dim=1)
        b = x * torch.log(x)
        b[torch.isnan(b)] = 0 # defining 0 log 0 = 0
        b = -1.0 * b.sum()
        return b


class Classifier(nn.Module):
    def __init__(self, args):
        super(Classifier, self).__init__()
        modules = [ nn.Dropout(p=0.2),
                    nn.ReLU(),
                    nn.Linear(args.hidden_size, args.hidden_size),
                    nn.Dropout(p=0.2),
                    nn.ReLU(),
                    nn.Linear(args.hidden_size, args.num_tasks)]
        if args.dataset_type == 'classification':
            modules.append(nn.Sigmoid()) # Jiang said sigmoiding here is fine
        self.net = nn.Sequential(*modules)

    def forward(self, x):
        return self.net(x)


class MOE(nn.Module):
    def __init__(self, args):
        super(MOE, self).__init__()
        self.args = args
        self.num_sources = args.num_sources
        self.classifiers = nn.ModuleList([Classifier(args) for _ in range(args.num_sources)])
        self.encoder = MPN(args)
        self.mmd = MMD(args)
        self.Us = nn.ParameterList([nn.Parameter(torch.zeros((args.hidden_size, args.m_rank)), requires_grad=True) for _ in range(args.num_sources)])
        #note zeros are replaced during initialization later
        if args.dataset_type == 'regression':
            self.mtl_criterion = nn.MSELoss(reduction='none')
            self.moe_criterion = nn.MSELoss(reduction='none')
        elif args.dataset_type == 'classification': #this half untested
            self.mtl_criterion = nn.BCELoss(reduction='none')
            self.moe_criterion = nn.BCELoss(reduction='none')
        self.entropy_criterion = HLoss()
        self.lambda_moe = args.lambda_moe
        self.lambda_critic = args.lambda_critic
        self.lambda_entropy = args.lambda_entropy

        self.domain_encs = None
    
    def mahalanobis_metric(self, p, mu, j):
        mahalanobis_distances_new = (p - mu).mm(self.Us[j].mm(self.Us[j].t())).mm((p - mu).t())
        mahalanobis_distances_new = mahalanobis_distances_new.diag().sqrt()
        return mahalanobis_distances_new
    
    def forward(self, smiles, features=None):
        encodings = self.encoder(smiles)
        classifier_outputs = [self.classifiers[i](encodings) for i in range(self.num_sources)]

        source_ids = range(self.num_sources)
        source_alphas = [-self.mahalanobis_metric(encodings, 
                            self.domain_encs[i], i).unsqueeze(0)
                            for i in source_ids]
        source_alphas = F.softmax(torch.cat(source_alphas, dim=0), dim=0) # n_source x bs
        output_moe = sum([ source_alphas[j].unsqueeze(1).repeat(1, 1) *
                            classifier_outputs[j] for j in source_ids])
        return output_moe
    
    def compute_domain_encs(self, all_train_smiles):
        domain_encs = []
        for i in range(len(all_train_smiles)):
            train_smiles = all_train_smiles[i]
            train_batches = []
            for j in range(0, len(train_smiles), self.args.batch_size):
                train_batches.append(train_smiles[j:j + self.args.batch_size])
            means_sum = torch.zeros(self.args.hidden_size)
            if self.args.cuda:
                means_sum = means_sum.cuda()
            for train_batch in train_batches:
                with torch.no_grad():
                    batch_encs = self.encoder(train_batch) #bs x hidden
                means_sum += torch.mean(batch_encs, dim=0)
            domain_encs.append(means_sum / len(train_batches))
        self.domain_encs = domain_encs
    
    def compute_minibatch_domain_encs(self, train_smiles):
        domain_encs = []
        for i in range(len(train_smiles)):
            train_batch = train_smiles[i]
            with torch.no_grad():
                batch_encs = self.encoder(train_batch) #bs x hidden
            domain_encs.append(torch.mean(batch_encs, dim=0))
        self.domain_encs = domain_encs
    
    def set_domain_encs(self, domain_encs):
        self.domain_encs = domain_encs
    
    def get_domain_encs(self):
        return self.domain_encs

    def compute_loss(self, train_smiles, train_targets, test_smiles):  # TODO(moe) parallelize?
        '''
        Computes and aggregates relevant losses for mixture of experts model. 
        :param train_smiles: n_sources x batch_size array of smiles
        :param train_targets: n_sources array of torch tensors of labels, each of dimension = batch_size
        :param test_smiles: batch_size array of smiles
        :return: a scalar representing aggregated losses for moe model
        '''
        if self.args.batch_domain_encs:
            self.compute_minibatch_domain_encs(train_smiles)
        encodings = [self.encoder(source_batch) for source_batch in train_smiles] # each bs x hs
        all_encodings = torch.cat(encodings, dim=0) # nsource*bs x hs
        
        classifier_outputs = [] # will be nsource x nsource of bs x hs
        for i in range(len(encodings)):
            outputs = []
            for j in range(len(self.classifiers)):
                outputs.append(self.classifiers[j](encodings[i]))
            classifier_outputs.append(outputs)
        supervised_outputs = torch.cat([classifier_outputs[i][i] for i in range(len(encodings))], dim=0)

        train_targets = [list(tt) for tt in train_targets]
        train_target_masks = []
        for i in range(len(train_targets)):
            train_target_masks.append(torch.Tensor([[x is not None for x in tb] for tb in train_targets[i]]))
            train_targets[i] = torch.Tensor([[0 if x is None else x for x in tb] for tb in train_targets[i]])
        if self.args.cuda:
            train_targets = [tl.cuda() for tl in train_targets]
            train_target_masks = [tlm.cuda() for tlm in train_target_masks]
        supervised_targets = torch.cat(train_targets, dim=0)
        supervised_mask = torch.cat(train_target_masks, dim=0)
        mtl_loss = self.mtl_criterion(supervised_outputs, supervised_targets)
        mtl_loss = mtl_loss * supervised_mask
        mtl_loss = mtl_loss.sum() / supervised_mask.sum()

        test_encodings = self.encoder(test_smiles)
        adv_loss = self.mmd(all_encodings, test_encodings)

        moe_loss = 0
        entropy_loss = 0
        source_ids = range(len(encodings))
        for i in source_ids:
            support_ids = [x for x in source_ids if x != i]
            support_alphas = [-self.mahalanobis_metric(encodings[i], 
                               self.domain_encs[j].detach(), j).unsqueeze(0)
                                for j in support_ids] # n_source-1 of 1 x bs
            support_alphas = F.softmax(torch.cat(support_alphas, dim=0), dim=0)

            source_alphas = [-self.mahalanobis_metric(encodings[i], 
                               self.domain_encs[j].detach(), j).unsqueeze(0)
                                for j in source_ids] # n_source of 1 x bs
            source_alphas = F.softmax(torch.cat(source_alphas, dim=0), dim=0) # n_source x bs

            output_moe_i = sum([ support_alphas[idx].unsqueeze(1).repeat(1, 1) *
                                 classifier_outputs[i][j] for idx, j in enumerate(support_ids)])

            moe_loss_i = self.moe_criterion(output_moe_i,
                                            train_targets[i])
            moe_loss_i = moe_loss_i * train_target_masks[i]
            if train_target_masks[i].sum() > 0:
                moe_loss_i = moe_loss_i.sum() / train_target_masks[i].sum()
                moe_loss += moe_loss_i
            entropy_loss += self.entropy_criterion(source_alphas)
        
        loss = (1.0 - self.lambda_moe) * mtl_loss + self.lambda_moe * moe_loss + self.lambda_critic * adv_loss + self.lambda_entropy * entropy_loss

        return loss
