import torch
import copy
import random
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler
import networkx as nx
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import kneighbors_graph
from .model_3M import Encoder_overall
from .preprocess_3M import adjacent_matrix_preprocessing
from .optimal_clustering_HLN import R5 


class Train_3M:
    def __init__(self,
                 data,
                 datatype='Triplet',
                 device=torch.device('cpu'),
                 random_seed=2024,
                 learning_rate=0.001,
                 weight_decay=2e-2,
                 epochs=200,
                 dim_input=3000,
                 dim_output=128,
                 weight_factors=[1, 3, 3],
                 Arg=None
                 ):
        '''\

        Parameters
        ----------
        data : dict
            dict object of spatial multi-omics data.
        datatype : string, optional
            Data type of input
            The default is 'Triplet'. To date, real-worlk triplet modality data is still unavailable. We define default data type as 'Triplet' temporarily.
        device : string, optional
            Using GPU or CPU? The default is 'cpu'.
        random_seed : int, optional
            Random seed to fix model initialization. The default is 2022.
        learning_rate : float, optional
            Learning rate for ST representation learning. The default is 0.001.
        weight_decay : float, optional
            Weight decay to control the influence of weight parameters. The default is 0.00.
        epochs : int, optional
            Epoch for model training. The default is 1500.
        dim_input : int, optional
            Dimension of input feature. The default is 3000.
        dim_output : int, optional
            Dimension of output representation. The default is 64.
        weight_factors : list, optional
            Weight factors to balance the influcences of different omics data on model training.

        Returns
        -------
        The learned representation 'self.emb_combined'.

        '''
        self.arg = Arg
        self.data = data.copy()
        self.datatype = datatype
        self.device = device
        self.random_seed = random_seed
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.dim_input = dim_input
        self.dim_output = dim_output
        self.weight_factors = weight_factors

        # adj
        self.adata_omics1 = self.data['adata_omics1']
        self.adata_omics2 = self.data['adata_omics2']
        self.adata_omics3 = self.data['adata_omics3']
        self.adj = adjacent_matrix_preprocessing(self.adata_omics1, self.adata_omics2, self.adata_omics3)
        self.adj_spatial_omics1 = self.adj['adj_spatial_omics1'].to(self.device).to_dense()
        self.adj_spatial_omics2 = self.adj['adj_spatial_omics2'].to(self.device).to_dense()
        self.adj_spatial_omics3 = self.adj['adj_spatial_omics3'].to(self.device).to_dense()
        self.adj_feature_omics1 = self.adj['adj_feature_omics1'].to(self.device).to_dense()
        self.adj_feature_omics2 = self.adj['adj_feature_omics2'].to(self.device).to_dense()
        self.adj_feature_omics3 = self.adj['adj_feature_omics3'].to(self.device).to_dense()

        self.adj_feature_omics1_copy = copy.deepcopy(self.adj_feature_omics1)
        self.adj_feature_omics2_copy = copy.deepcopy(self.adj_feature_omics2)
        self.adj_feature_omics3_copy = copy.deepcopy(self.adj_feature_omics3)

        self.paramed_adj_omics1 = Parametered_Graph(self.adj_feature_omics1, self.device).to(self.device)
        self.paramed_adj_omics2 = Parametered_Graph(self.adj_feature_omics2, self.device).to(self.device)
        self.paramed_adj_omics3 = Parametered_Graph(self.adj_feature_omics3, self.device).to(self.device)

        self.clustering = R5(self.datatype, self.arg)

        self.EMA_coeffi = 0.9

        # feature
        self.features_omics1 = torch.FloatTensor(self.adata_omics1.obsm['feat'].copy()).to(self.device)
        self.features_omics2 = torch.FloatTensor(self.adata_omics2.obsm['feat'].copy()).to(self.device)
        self.features_omics3 = torch.FloatTensor(self.adata_omics3.obsm['feat'].copy()).to(self.device)

        # dimension of input feature
        self.dim_input1 = self.features_omics1.shape[1]
        self.dim_input2 = self.features_omics2.shape[1]
        self.dim_input3 = self.features_omics3.shape[1]
        self.dim_output1 = self.dim_output
        self.dim_output2 = self.dim_output
        self.dim_output3 = self.dim_output

    def train(self):
        self.model = Encoder_overall(self.dim_input1, self.dim_output1, self.dim_input2, self.dim_output2,
                                     self.dim_input3, self.dim_output3).to(self.device)
        self.optimizer = torch.optim.SGD(list(self.model.parameters()) +
                                         list(self.paramed_adj_omics1.parameters()) +
                                         list(self.paramed_adj_omics2.parameters()) +
                                         list(self.paramed_adj_omics3.parameters()),
                                         lr=self.learning_rate,
                                         momentum=0.9,
                                         weight_decay=self.weight_decay
                                        )

        self.model.train()
        for epoch in tqdm(range(self.epochs)):
            self.model.train()
            results = self.model(self.features_omics1, self.features_omics2, self.features_omics3,
                                 self.adj_spatial_omics1, self.adj_feature_omics1,
                                 self.adj_spatial_omics2, self.adj_feature_omics2,
                                 self.adj_spatial_omics3, self.adj_feature_omics3)

            # reconstruction loss
            self.loss_recon_omics1 = F.mse_loss(self.features_omics1, results['emb_recon_omics1'])
            self.loss_recon_omics2 = F.mse_loss(self.features_omics2, results['emb_recon_omics2'])
            self.loss_recon_omics3 = F.mse_loss(self.features_omics3, results['emb_recon_omics3'])

            recon_loss = (self.weight_factors[0] * self.loss_recon_omics1 + self.weight_factors[1] * self.loss_recon_omics2 + \
                   self.weight_factors[2] * self.loss_recon_omics3)

            ##-------------------------------------> cal graph loss begin<---------------------------------------------------

            dpcl_loss = self.clustering(results['emb_latent_combined'], epoch)

            # update graph

            updated_adj_omics1 = self.paramed_adj_omics1.normalize()
            updated_adj_omics2 = self.paramed_adj_omics2.normalize()
            updated_adj_omics3 = self.paramed_adj_omics3.normalize()

            loss_adj = (torch.norm(updated_adj_omics1 - self.adj_feature_omics1_copy.detach(), p='fro') +
                        torch.norm(updated_adj_omics2 - self.adj_feature_omics2_copy.detach(), p='fro') +
                        torch.norm(updated_adj_omics3 - self.adj_feature_omics3_copy.detach(), p='fro')) / 3

            loss = recon_loss + dpcl_loss + loss_adj

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            self.adj_feature_omics1 = self.paramed_adj_omics1()
            self.adj_feature_omics2 = self.paramed_adj_omics2()
            self.adj_feature_omics3 = self.paramed_adj_omics3()

            self.adj_feature_omics1_copy = self.EMA_coeffi * self.adj_feature_omics1_copy + (
                    1 - self.EMA_coeffi) * updated_adj_omics1.detach().clone()
            self.adj_feature_omics2_copy = self.EMA_coeffi * self.adj_feature_omics2_copy + (
                    1 - self.EMA_coeffi) * updated_adj_omics2.detach().clone()
            self.adj_feature_omics3_copy = self.EMA_coeffi * self.adj_feature_omics3_copy + (
                    1 - self.EMA_coeffi) * updated_adj_omics3.detach().clone()

        print("Model training finished!\n")

        with torch.no_grad():
            self.model.eval()
            results = self.model(self.features_omics1, self.features_omics2, self.features_omics3,
                                 self.adj_spatial_omics1, self.adj_feature_omics1,
                                 self.adj_spatial_omics2, self.adj_feature_omics2,
                                 self.adj_spatial_omics3, self.adj_feature_omics3
                                 )

        emb_omics1 = F.normalize(results['emb_latent_omics1'], p=2, eps=1e-12, dim=1)
        emb_omics2 = F.normalize(results['emb_latent_omics2'], p=2, eps=1e-12, dim=1)
        emb_omics3 = F.normalize(results['emb_latent_omics3'], p=2, eps=1e-12, dim=1)
        emb_combined = F.normalize(results['emb_latent_combined'], p=2, eps=1e-12, dim=1)

        output = {'emb_latent_omics1': emb_omics1.detach().cpu().numpy(),
                  'emb_latent_omics2': emb_omics2.detach().cpu().numpy(),
                  'emb_latent_omics3': emb_omics3.detach().cpu().numpy(),
                  'PRAGA': emb_combined.detach().cpu().numpy(),
                  }

        return output


class Parametered_Graph(nn.Module):
    def __init__(self, adj, device):
        super(Parametered_Graph, self).__init__()
        self.adj = adj
        self.device = device

        n = self.adj.shape[0]
        self.paramed_adj_omics = nn.Parameter(torch.FloatTensor(n, n))
        self.paramed_adj_omics.data.copy_(self.adj)

    def forward(self):
        adj = (self.paramed_adj_omics + self.paramed_adj_omics.t()) / 2
        adj = nn.ReLU(inplace=True)(adj)
        adj = self._normalize(adj.to(self.device) + torch.eye(adj.shape[0]).to(self.device))
        return adj.to(self.device)

    def normalize(self, A=None):

        if A is None:
            adj = (self.paramed_adj_omics + self.paramed_adj_omics.t()) / 2
            adj = nn.ReLU(inplace=True)(adj)
            normalized_adj = self._normalize(adj.to(self.device) + torch.eye(adj.shape[0]).to(self.device))
        else:
            adj = (A + A.t()) / 2
            adj = nn.ReLU(inplace=True)(adj)
            normalized_adj = self._normalize(adj.to(self.device) + torch.eye(adj.shape[0]).to(self.device))
        return normalized_adj

    def _normalize(self, mx):
        rowsum = mx.sum(1)
        r_inv = rowsum.pow(-1 / 2).flatten()
        r_inv[torch.isinf(r_inv)] = 0.
        r_mat_inv = torch.diag(r_inv)
        mx = r_mat_inv @ mx
        mx = mx @ r_mat_inv
        return mx

