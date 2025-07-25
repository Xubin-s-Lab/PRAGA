import  torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch.optim as optim
from torch.autograd import Variable
from sklearn.neighbors import kneighbors_graph
from torch.distributions.multivariate_normal import MultivariateNormal
from sklearn.mixture import GaussianMixture
from clustering_utils import split_and_merge_op, pairwise_distance

class R5(nn.Module):
    def __init__(self, data_type=None, arg=None):
        super(R5, self).__init__()
        self.tau = arg.tau
        self.N = 2
        self.begin = False
        self.datatype = data_type
        self.device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
        
        self.weight = 1
        self.warmup = 99
        self.stage = 100
        self.arg = Arg(init_K=5)
        self.arg.prior_nu = 100

    def forward(self, feat, epoch):

        if epoch>self.warmup:
            if (self.begin == False) or (epoch % self.stage == 0):
                print('updating clustring...')
                self.begin = True
                self.centroids = split_and_merge_op(feat, self.arg).to(self.device)
                dist = pairwise_distance(feat, self.centroids)
                value, pred = torch.min(dist, dim=1)
            else:
                dist = pairwise_distance(feat, self.centroids)
                value, pred = torch.min(dist, dim=1)

            cl_loss = self.CL(feat, pred, self.centroids)
        else:
            cl_loss = 0
        return cl_loss * self.weight

    def get_pred(self, x, class_num):
        gmm = GaussianMixture(n_components=class_num, random_state=0)
        gmm.fit(x)
        labels = gmm.predict(x)
        return labels

    def CL(self, feat, pred, centers):
        loss = 0
        for label, center in enumerate(centers):
            center = center.float()
            feat = feat.float()
            if feat[pred == label].shape[0] > 0 and feat[pred != label].shape[0] > 0:

                pos = torch.div(
                    torch.matmul(feat[pred == label], center.unsqueeze(0).T),
                    self.tau)
                neg = torch.div(
                    torch.matmul(feat, center.unsqueeze(0).T),
                    self.tau)

                pos = torch.mean(torch.exp(pos.squeeze()))
                neg = torch.mean(torch.exp(neg.squeeze()))

                loss = loss - torch.log(pos / neg)

        return loss / centers.shape[0]


class Arg:
    def __init__(self, init_K):
        self.pi_prior = 'uniform'
        self.prior_dir_counts = 0.1
        self.prior_mu_0 = 'data_mean'
        self.prior_sigma_choice = 'isotropic'
        self.prior_sigma_scale = .005
        self.prior_kappa = 0.0001
        self.prior_nu = init_K * 3
        self.class_num = init_K
        self.temperature = 1


class uniform_loss(nn.Module):
    def __init__(self, t=0.07):
        super(uniform_loss, self).__init__()
        self.t = t

    def forward(self, x):
        return x.matmul(x.T).div(self.t).exp().sum(dim=-1).log().mean()








