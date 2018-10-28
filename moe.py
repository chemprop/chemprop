import torch
import torch.nn as nn
import torch.nn.functional as F

from mpn import MPN

def compute_pairwise_distances(x, y):
    '''
    Computes the squared pairwise Euclidean distances between x and y.
    :param x: a tensor of shape [num_x_samples, num_features]
    :param y: a tensor of shape [num_y_samples, num_features]
    :return: a distance matrix of dimensions [num_x_samples, num_y_samples]
    Raise:
        ValueError if the inputs do not match the specified dimensions
    '''
    print('In compute_pairwise_distances  NOTE THAT THIS DOES NOT SEEM TO BE WORKING.')
    print('x and y should have dimensions num_i_samples x num_features, it seems like the first dimension is number of atoms instead')
    print('Uncomment print statements in utils/compute_pairwise_distances to observe this')
    '''
    print('x.size() =', x.size())
    print('x.size()[1] =', x.size()[1])
    print('y.size() =', y.size())
    print('y.size()[1] =', y.size()[1])
    '''
    if not len(x.size()) == len(y.size()) == 2:
        raise ValueError('Both inputs should be matrices.')
    if x.size()[1] != y.size()[1]:
        raise ValueError('The number of features should be the same.')

    # By making the 'inner' dimensions of the two matrices equal to 1 using broadcasting then we are essentially
    # subtracting every pair of rows of x and y
    norm = lambda x:torch.sum(x * x, 1)
    return norm(x.unsqueeze(2) - y.t())

def gaussian_kernel_matrix(x, y, sigmas):
    '''
    Computes a gaussian RBK between the samples of x and y
    We create a sum of multiple gaussian kernels each having a width sigma_i
    :param x: a tensor of shape [num_samples, num_features]
    :param y: a tensor of shape [num_samples, num_features]
    :param sigmas: a tensor of floats which denote the widths of each of the gaussians in the kernel
    :return: a tensor of shape [num_samples{x}, num_samples{y}] with the RBF kernel
    '''
    beta = 1. / (2. * (sigmas.unsqueeze(1)))
    dist = compute_pairwise_distances(x, y)
    s = torch.matmul(beta, dist.view(1, -1))

    return(torch.sum(torch.exp(-s), 0)).view_as(dist)

def maximum_mean_discrepancy(x, y, kernel=gaussian_kernel_matrix):
    '''
    Computes the Maximum Mean Discrepancy (MMD) of two samples: x and y
    Maximum Mean Discrepancy (MMD) is a distance-measure between the samples of the distributions of x and y. Here we
    kernel two sample estimate using the empirical mean of the two distributions
    :param x: a tensor of shape [num_samples, num_features]
    :param y: a tensor of shape [num_samples, num_features]
    :param kernel: a function which computes the kernel in MMD.  Defaults to the GaussianKernelMatrix.
    :return: a scalar denoting the squared maximum mean discrepancy loss
    '''

    cost = torch.mean(kernel(x, x))
    cost += torch.mean(kernel(y, y))
    cost -= 2 * torch.mean(kernel(x, y))

    cost = torch.clamp(cost, min=0)
    return cost

class HLoss(nn.Module):
    def __init__(self):
        super(HLoss, self).__init__()

    def forward(self, x):
        #b = F.softmax(x, dim=1) * F.log_softmax(x, dim=1)
        b = x * torch.log(x)
        b = -1.0 * b.sum()
        return b

class Classifier(torch.nn.Module):
    def __init__(self, args):
        super(Classifier, self).__init__()
        self.net = nn.Sequential(
                                nn.Dropout(p=0.2),
                                nn.ReLU(),
                                nn.Linear(args.hidden_size, args.hidden_size),
                                nn.Dropout(p=0.2),
                                nn.ReLU(),
                                nn.Linear(args.hidden_size, args.num_tasks)
                            )
    def forward(self, x):
        return self.net(x)

class MOE(nn.Module):
    def __init__(self, args):
        super(MOE, self).__init__()
        self.args = args
        self.num_sources = args.num_sources
        self.classifiers = nn.ModuleList([Classifier(args) for _ in range(args.num_sources)])
        self.encoder = MPN(args)
        self.Us = nn.ParameterList([torch.FloatTensor(args.hidden_size, args.m_rank) for _ in range(args.num_sources)])
        self.mtl_criterion = nn.L1Loss()
        self.moe_criterion = nn.L1Loss()
        self.entropy_criterion = HLoss()
        self.lambda_moe = args.lambda_moe
        self.lambda_critic = args.lambda_critic
        self.lambda_entropy = args.lambda_entropy
    
    def mahalanobis_metric(self, p, mu, j):

        mahalanobis_distances_new = (p - mu).mm(self.Us[j].mm(self.Us[j].t())).mm((p - mu).t())
        mahalanobis_distances_new = mahalanobis_distances_new.diag().sqrt()
        return mahalanobis_distances_new.detach() #TODO check is this detach correct? and check dims
    
    def forward(self, smiles):
        encodings = self.encoder(smiles)
        classifier_outputs = [self.classifiers[i](smiles) for i in range(self.num_sources)]

        source_ids = range(self.num_sources)
        source_alphas = [-self.mahalanobis_metric(encodings, 
                            self.domain_encs[i], i)
                            for i in source_ids]
        source_alphas = F.softmax(source_alphas)

        output_moe = sum([ alpha.unsqueeze(1).repeat(1, 1) *
                                F.softmax(classifier_outputs[id], dim=1)
                        for alpha, id in zip(source_alphas, source_ids)])
        return output_moe
    
    def compute_domain_encs(self, all_train_smiles):
        # domain_encoding in jiang's code, if i read it correctly. should be called at beginning of each epoch
        domain_encs = []
        for i in range(len(all_train_smiles)):
            batch_encs = [self.encoder(all_train_smiles[i][j]) for j in range(len(all_train_smiles[i]))]
            means = [torch.mean(batch_encs[i], dim=0, keepdim=True) for i in range(len(batch_encs))]
            domain_encs.append(torch.means(torch.cat(means, dim=0), dim=0))
        self.domain_encs = domain_encs

    def compute_loss(self, train_smiles, train_labels, test_smiles): #TODO parallelize?
        '''
        Computes and aggregates relevant losses for mixture of experts model. 
        :param train_smiles: n_sources x batch_size array of smiles
        :param train_labels: n_sources x batch_size array of labels, as torch tensors
        :param test_smiles: batch_size array of smiles
        :return: a scalar representing aggregated losses for moe model
        '''
        encodings = [self.encoder(train_smiles) for _ in range(len(train_smiles))] # each bs x hs
        all_encodings = torch.cat(encodings, dim=0) # nsource*bs x hs
        
        classifier_outputs = [] # will be nsource x nsource of bs x hs
        for i in range(len(encodings)):
            outputs = []
            for j in range(len(self.classifiers)):
                outputs.append(self.classifiers[j](encodings[i]))
            classifier_outputs.append(outputs)
        supervised_outputs = torch.cat([classifier_outputs[i][i] for i in range(len(encodings))], dim=0)
        mtl_loss = self.mtl_criterion(supervised_outputs, torch.cat([train_labels], dim=0))
        
        test_encodings = self.encoder(test_smiles)
        adv_loss = maximum_mean_discrepancy(all_encodings, test_encodings)
        
        moe_loss = 0
        entropy_loss = 0
        source_ids = range(len(encodings))
        for i in source_ids:
            support_ids = [x for x in source_ids if x != i]
            support_alphas = [-self.mahalanobis_metric(encodings[i], 
                               self.domain_encs[j], j)
                                for j in support_ids]
            support_alphas = F.softmax(support_alphas)

            source_alphas = [-self.mahalanobis_metric(encodings[i], 
                               self.domain_encs[j], j)
                                for j in source_ids]
            source_alphas = F.softmax(source_alphas)

            output_moe_i = sum([ alpha.unsqueeze(1).repeat(1, 1) *
                                 F.softmax(classifier_outputs[i][id], dim=1)
                           for alpha, id in zip(support_alphas, support_ids) ])
            moe_loss += self.moe_criterion(torch.log(output_moe_i),
                            train_labels[i])
            source_alphas = torch.stack(source_alphas, dim=0)
            entropy_loss += self.entropy_criterion(source_alphas)
        
        loss = (1.0 - self.lambda_moe) * mtl_loss + self.lambda_moe * moe_loss + self.lambda_critic * adv_loss + self.lambda_entropy * entropy_loss
        return loss