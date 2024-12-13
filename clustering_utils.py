import numpy as np
import copy
import random
import torch.nn.functional as F
from sklearn.utils._joblib import Parallel, delayed, effective_n_jobs
from sklearn.utils import check_random_state
from sklearn.neighbors import NearestNeighbors
from sklearn.mixture import GaussianMixture
import torch
from tqdm import tqdm

def pairwise_distance(data1, data2, batch_size=None):
    r'''
    using broadcast mechanism to calculate pairwise ecludian distance of data
    the input data is N*M matrix, where M is the dimension
    we first expand the N*M matrix into N*1*M matrix A and 1*N*M matrix B
    then a simple elementwise operation of A and B will handle the pairwise operation of points represented by data
    '''


    if batch_size == None:

        # N*1*M
        A = data1.unsqueeze(dim=1)

        # 1*N*M
        B = data2.unsqueeze(dim=0)


        dis = (A-B)**2
        #return N*N matrix for pairwise distance
        dis = dis.sum(dim=-1)
        # #  torch.cuda.empty_cache()
    else:
        # N*1*M
        A = data1.unsqueeze(dim=1)

        # 1*N*M
        B = data2.unsqueeze(dim=0)
        i = 0
        dis = torch.zeros(data1.shape[0], data2.shape[0])
        while i < data1.shape[0]:
            if(i+batch_size < data1.shape[0]):
                dis_batch = (A[i:i+batch_size]-B)**2
                dis_batch = dis_batch.sum(dim=-1)
                dis[i:i+batch_size] = dis_batch
                i = i+batch_size
                #  torch.cuda.empty_cache()
            elif(i+batch_size >= data1.shape[0]):
                dis_final = (A[i:] - B)**2
                dis_final = dis_final.sum(dim=-1)
                dis[i:] = dis_final
                #  torch.cuda.empty_cache()
                break
    #  torch.cuda.empty_cache()
    return dis

def cluster_acc(y_true, y_pred, return_ind=False):
    """
    Calculate clustering accuracy. Require scikit-learn installed

    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`

    # Return
        accuracy, in [0,1]
    """
    # __import__("ipdb").set_trace()
    y_true = y_true.astype(int)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=int)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1

    ind = linear_assignment(w.max() - w)
    ind = np.vstack(ind).T

    if return_ind:
        return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size, ind, w
    else:
        return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size

# class K_Means:
#
#     def __init__(self, k=3, tolerance=1e-4, max_iterations=100, init='k-means++',
#                  n_init=10, random_state=None, n_jobs=None, pairwise_batch_size=None, mode=None):
#         self.k = k
#         self.tolerance = tolerance
#         self.max_iterations = max_iterations
#         self.init = init
#         self.n_init = n_init
#         self.random_state = random_state
#         self.n_jobs = n_jobs
#         self.pairwise_batch_size = pairwise_batch_size
#         self.mode = mode
#
#     def split_for_val(self, l_feats, l_targets, val_prop=0.2):
#
#         np.random.seed(0)
#
#         # Reserve some labelled examples for validation
#         num_val_instances = int(val_prop * len(l_targets))
#         val_idxs = np.random.choice(range(len(l_targets)), size=(num_val_instances), replace=False)
#         val_idxs.sort()
#         remaining_idxs = list(set(range(len(l_targets))) - set(val_idxs.tolist()))
#         remaining_idxs.sort()
#         remaining_idxs = np.array(remaining_idxs)
#
#         val_l_targets = l_targets[val_idxs]
#         val_l_feats = l_feats[val_idxs]
#
#         remaining_l_targets = l_targets[remaining_idxs]
#         remaining_l_feats = l_feats[remaining_idxs]
#
#         return remaining_l_feats, remaining_l_targets, val_l_feats, val_l_targets
#
#
#     def kpp(self, X, pre_centers=None, k=10, random_state=None):
#         random_state = check_random_state(random_state)
#
#         if pre_centers is not None:
#
#             C = pre_centers
#
#         else:
#
#             C = X[random_state.randint(0, len(X))]
#
#         C = C.view(-1, X.shape[1])
#
#         while C.shape[0] < k:
#
#             dist = pairwise_distance(X, C, self.pairwise_batch_size)
#             dist = dist.view(-1, C.shape[0])
#             d2, _ = torch.min(dist, dim=1)
#             prob = d2/d2.sum()
#             cum_prob = torch.cumsum(prob, dim=0)
#             r = random_state.rand()
#
#             if len((cum_prob >= r).nonzero()) == 0:
#                 debug = 0
#             else:
#                 ind = (cum_prob >= r).nonzero()[0][0]
#             C = torch.cat((C, X[ind].view(1, -1)), dim=0)
#
#         return C
#
#
#     def fit_once(self, X, random_state):
#
#         X = X.cuda()
#
#         centers = torch.zeros(self.k, X.shape[1]).type_as(X).cuda()
#         labels = -torch.ones(len(X))
#         #initialize the centers, the first 'k' elements in the dataset will be our initial centers
#
#         if self.init == 'k-means++':
#             centers = self.kpp(X, k=self.k, random_state=random_state)
#
#         elif self.init == 'random':
#
#             random_state = check_random_state(self.random_state)
#             idx = random_state.choice(len(X), self.k, replace=False)
#             for i in range(self.k):
#                 centers[i] = X[idx[i]]
#
#         else:
#             for i in range(self.k):
#                 centers[i] = X[i]
#
#         #begin iterations
#
#         best_labels, best_inertia, best_centers = None, None, None
#         for i in range(self.max_iterations):
#
#             centers_old = centers.clone()
#             dist = pairwise_distance(X, centers, self.pairwise_batch_size)
#             mindist, labels = torch.min(dist, dim=1)
#             inertia = mindist.sum()
#
#
#
#             for idx in range(self.k):
#                 selected = torch.nonzero(labels == idx).squeeze().cuda()
#                 selected = torch.index_select(X, 0, selected)
#                 centers[idx] = selected.mean(dim=0)
#
#             if best_inertia is None or inertia < best_inertia:
#                 best_labels = labels.clone()
#                 best_centers = centers.clone()
#                 best_inertia = inertia
#
#             center_shift = torch.sum(torch.sqrt(torch.sum((centers - centers_old) ** 2, dim=1)))
#             if center_shift ** 2 < self.tolerance:
#                 #break out of the main loop if the results are optimal, ie. the centers don't change their positions much(more than our tolerance)
#                 break
#
#         return best_labels, best_inertia, best_centers, i + 1
#
#
#     def fit_mix_once(self, u_feats, l_feats, l_targets, random_state):
#
#         def supp_idxs(c):
#             return l_targets.eq(c).nonzero().squeeze(1)
#
#         l_classes = torch.unique(l_targets)
#         support_idxs = list(map(supp_idxs, l_classes))
#         l_centers = torch.stack([l_feats[idx_list].mean(0) for idx_list in support_idxs])
#         cat_feats = torch.cat((l_feats, u_feats))
#
#         centers = torch.zeros([self.k, cat_feats.shape[1]]).type_as(cat_feats)
#         centers[:len(l_classes)] = l_centers
#
#         labels = -torch.ones(len(cat_feats)).type_as(cat_feats).long()
#
#         l_classes = l_classes.cpu().long().numpy()
#         l_targets = l_targets.cpu().long().numpy()
#         l_num = len(l_targets)
#         cid2ncid = {cid:ncid for ncid, cid in enumerate(l_classes)}  # Create the mapping table for New cid (ncid)
#         for i in range(l_num):
#             labels[i] = cid2ncid[l_targets[i]]
#
#         #initialize the centers, the first 'k' elements in the dataset will be our initial centers
#         centers = self.kpp(u_feats, l_centers, k=self.k, random_state=random_state)
#
#         # Begin iterations
#         best_labels, best_inertia, best_centers = None, None, None
#         for it in range(self.max_iterations):
#             centers_old = centers.clone()
#
#             dist = pairwise_distance(u_feats, centers, self.pairwise_batch_size)
#             u_mindist, u_labels = torch.min(dist, dim=1)
#             u_inertia = u_mindist.sum()
#             l_mindist = torch.sum((l_feats - centers[labels[:l_num]])**2, dim=1)
#             l_inertia = l_mindist.sum()
#             inertia = u_inertia + l_inertia
#             labels[l_num:] = u_labels
#
#             for idx in range(self.k):
#
#                 selected = torch.nonzero(labels == idx).squeeze()
#                 selected = torch.index_select(cat_feats, 0, selected)
#                 centers[idx] = selected.mean(dim=0)
#
#             if best_inertia is None or inertia < best_inertia:
#                 best_labels = labels.clone()
#                 best_centers = centers.clone()
#                 best_inertia = inertia
#
#             center_shift = torch.sum(torch.sqrt(torch.sum((centers - centers_old) ** 2, dim=1)))
#
#             if center_shift ** 2 < self.tolerance:
#                 #break out of the main loop if the results are optimal, ie. the centers don't change their positions much(more than our tolerance)
#                 break
#
#         return best_labels, best_inertia, best_centers, i + 1
#
#
#     def fit(self, X):
#         random_state = check_random_state(self.random_state)
#         best_inertia = None
#         if effective_n_jobs(self.n_jobs) == 1:
#             for it in range(self.n_init):
#                 labels, inertia, centers, n_iters = self.fit_once(X, random_state)
#                 if best_inertia is None or inertia < best_inertia:
#                     self.labels_ = labels.clone()
#                     self.cluster_centers_ = centers.clone()
#                     best_inertia = inertia
#                     self.inertia_ = inertia
#                     self.n_iter_ = n_iters
#         else:
#             # parallelisation of k-means runs
#             seeds = random_state.randint(np.iinfo(np.int32).max, size=self.n_init)
#             results = Parallel(n_jobs=self.n_jobs, verbose=0)(delayed(self.fit_once)(X, seed) for seed in seeds)
#             # Get results with the lowest inertia
#             labels, inertia, centers, n_iters = zip(*results)
#             best = np.argmin(inertia)
#             self.labels_ = labels[best]
#             self.inertia_ = inertia[best]
#             self.cluster_centers_ = centers[best]
#             self.n_iter_ = n_iters[best]
#
#
#     def fit_mix(self, u_feats, l_feats, l_targets):
#
#         random_state = check_random_state(self.random_state)
#         best_inertia = None
#         fit_func = self.fit_mix_once
#
#         if effective_n_jobs(self.n_jobs) == 1:
#             for it in range(self.n_init):
#
#                 labels, inertia, centers, n_iters = fit_func(u_feats, l_feats, l_targets, random_state)
#
#                 if best_inertia is None or inertia < best_inertia:
#                     self.labels_ = labels.clone()
#                     self.cluster_centers_ = centers.clone()
#                     best_inertia = inertia
#                     self.inertia_ = inertia
#                     self.n_iter_ = n_iters
#
#         else:
#
#             # parallelisation of k-means runs
#             seeds = random_state.randint(np.iinfo(np.int32).max, size=self.n_init)
#             results = Parallel(n_jobs=self.n_jobs, verbose=0)(delayed(fit_func)(u_feats, l_feats, l_targets, seed)
#                                                               for seed in seeds)
#             # Get results with the lowest inertia
#
#             labels, inertia, centers, n_iters = zip(*results)
#             best = np.argmin(inertia)
#             self.labels_ = labels[best]
#             self.inertia_ = inertia[best]
#             self.cluster_centers_ = centers[best]
#             self.n_iter_ = n_iters[best]


def get_sub_assign_with_one_cluster(feat, labels, k, prior):
    counts = []
    class_indices = labels == k
    class_sub_feat = feat[class_indices]

    if len(class_sub_feat) <= 2:
        c = torch.tensor([0, len(class_sub_feat)])
        class_sub_assign = torch.ones(len(class_sub_feat), dtype=torch.long)
        mu_subs = torch.mean(class_sub_feat, dim=0, keepdim=True)
        mu_subs = torch.cat([torch.zeros_like(mu_subs), mu_subs], dim=0)
        # NOTE: empty sub clusters
    else:

        gmm = GaussianMixture(n_components=2, random_state=0)
        gmm.fit(class_sub_feat)

        mu_subs = gmm.means_

        # 获取每个样本的聚类分配
        class_sub_assign = gmm.predict(class_sub_feat)

        class_sub_assign = torch.tensor(class_sub_assign)
        mu_subs = torch.tensor(mu_subs)

        # km = K_Means(k=2, init='k-means++', random_state=1, n_jobs=None, pairwise_batch_size=None)
        # km.fit(class_sub_feat)
        # class_sub_assign = km.labels_.cpu()
        # print(class_sub_assign.shape)
        # assert 0
        # mu_subs = km.cluster_centers_
        _, c = torch.unique(class_sub_assign, return_counts=True)
    counts.extend(c.cpu().numpy().tolist())

    data_covs_sub = compute_data_covs_hard_assignment(class_sub_assign, class_sub_feat, 2, mu_subs.cpu(), prior)

    # update prior
    mu_subs = prior.compute_post_mus(torch.tensor(counts), mu_subs.cpu())
    covs_sub = []
    for k in range(2):
        covs_sub_k = prior.compute_post_cov(counts[k], class_sub_feat[class_sub_assign == k].mean(axis=0),
                                            data_covs_sub[k])
        covs_sub.append(covs_sub_k)
    covs_sub = torch.stack(covs_sub)

    pi_sub = torch.tensor(counts) / float(len(class_sub_feat))
    return mu_subs, covs_sub, pi_sub, class_sub_assign

def get_sub_cluster_with_sskmeans(u_feat, labels, prior, args, ):
    # NOTE: reads cluster assignments from sskmeans, perform clustering within each cluster
    sub_clusters = []
    mu_sub_list = []
    cov_sub_list = []
    pi_sub_list = []
    class_sub_assign_list = []
    # only the unlabelled data will be splited or merged
    # l_labels = labels[:len(l_targets)].unique().cpu()
    # all_labels = labels[len(l_targets):].unique().cpu()

    for class_label in tqdm(labels.cpu().numpy().tolist()):
        mu_sub, cov_sub, pi_sub, class_sub_assign = get_sub_assign_with_one_cluster(u_feat, labels,
                                                                                    class_label, prior)
        # print(mu_sub.shape)
        # print(cov_sub.shape)
        # print(pi_sub.shape)
        sub_clusters.append([class_label, (mu_sub, cov_sub, pi_sub, class_sub_assign)])
        mu_sub_list.append(mu_sub)
        cov_sub_list.append(cov_sub)
        pi_sub_list.append(pi_sub)
        class_sub_assign_list.append(class_sub_assign)
        # print(class_sub_assign.shape)
        # assert 0
    return sub_clusters, mu_sub_list, cov_sub_list, pi_sub_list, class_sub_assign_list


# def split_rule(feats, sub_assignment, prior, mu, mu_subs):
def split_rule(class_feat, mus, feat_sub_0, feat_sub_1, mu_sub_0, mu_sub_1, num_sub_0, num_sub_1, prior):
    # NOTE: deal with empty clusters first, pi_sub is [0, 1], no split
    """
    feats: NxD, subset of features
    sub_assignment: N, 0 and 1 assignments
    mu: 1xD, cluster center
    mu_subs: 2xD, sub cluster centers
    return [k, bool], split the k-th cluster or not
    """
    # if len(feats) <= 5:
    #     # small clusters will not be splited
    #     return False
    #
    # if len(feats[sub_assignment == 0]) <= 5 or len(feats[sub_assignment == 1]) <= 5:
    #     return False

    log_ll_k = prior.log_marginal_likelihood(class_feat, mus)  ## k类内的 Nk 个 feats 和 mu
    log_ll_k1 = prior.log_marginal_likelihood(feat_sub_0, mu_sub_0) ## k类内的第一个子类的 Nk_1 个 feats 和 mu
    log_ll_k2 = prior.log_marginal_likelihood(feat_sub_1, mu_sub_1) ## k类内的第一个子类的 Nk_2 个 feats 和 mu
    N_k_1 = num_sub_0  ## 第一个子类的数量
    N_k_2 = num_sub_1  ## 第二个子类的数量

    return log_Hastings_ratio_split(1.0, torch.tensor(N_k_1), torch.tensor(N_k_2), log_ll_k1, log_ll_k2, log_ll_k, split_prob=0.1)


def merge_rule(mu1, cov1, pi1, mu2, cov2, pi2, feat1, feat2, prior=None):
    all_feat = torch.cat([feat1, feat2], dim=0)
    N_k_1 = feat1.shape[0]
    N_k_2 = feat2.shape[0]
    N_k = feat1.shape[0] + feat2.shape[0]

    if N_k > 0:
        mus_mean = (N_k_1 / N_k) * mu1 + (N_k_2 / N_k) * mu2
    else:
        # in case both are empty clusters
        mus_mean = torch.mean(torch.stack([mu1, mu2]), axis=0)
    if prior is None:
        raise NotImplementedError
    else:
        log_ll_k = prior.log_marginal_likelihood(all_feat, mus_mean)
        log_ll_k_1 = prior.log_marginal_likelihood(feat1, mu1)
        log_ll_k_2 = prior.log_marginal_likelihood(feat2, mu2)

    return log_Hastings_ratio_merge(1.0, N_k_1, N_k_2, log_ll_k_1, log_ll_k_2, log_ll_k, merge_prob=0.1)

def compute_data_covs_hard_assignment(labels, codes, K, mus, prior):
    # assume to be NIW prior
    covs = []
    for k in range(K):
        codes_k = codes[labels == k]
        N_k = float(len(codes_k))
        if N_k > 0:
            cov_k = torch.matmul(
                (codes_k - mus[k].cpu().repeat(len(codes_k), 1)).T,
                (codes_k - mus[k].cpu().repeat(len(codes_k), 1)),
            )
            cov_k = cov_k / N_k
        else:
            # NOTE: deal with empty cluster
            cov_k = torch.eye(codes.shape[1]) * 0.0005
        covs.append(cov_k)
    return torch.stack(covs)

def log_Hastings_ratio_split(
    alpha, N_k_1, N_k_2, log_ll_k_1, log_ll_k_2, log_ll_k, split_prob
):
    """This function computes the log Hastings ratio for a split.

    Args:
        alpha ([float]): The alpha hyperparameter
        N_k_1 ([int]): Number of points assigned to the first subcluster
        N_k_2 ([int]): Number of points assigned to the second subcluster
        log_ll_k_1 ([float]): The log likelihood of the points in the first subcluster
        log_ll_k_2 ([float]): The log likelihood of the points in the second subcluster
        log_ll_k ([float]): The log likelihood of the points in the second subcluster
        split_prob ([type]): Probability to split a cluster even if the Hastings' ratio is not > 1

        Returns a boolean indicating whether to perform a split
    """
    N_k = N_k_1 + N_k_2
    if N_k_2 > 0 and N_k_1 > 0:
        # each subcluster is not empty
        H = (
            np.log(alpha) + lgamma(N_k_1) + log_ll_k_1 + lgamma(N_k_2) + log_ll_k_2
        ) - (lgamma(N_k) + log_ll_k)
        split_prob = split_prob or torch.exp(H)
    else:
        H = torch.zeros(1)
        split_prob = 0

    # if Hastings ratio > 1 (or 0 in log space) perform split, if not, toss a coin
    return bool(H > 0)

def log_Hastings_ratio_merge(
    alpha, N_k_1, N_k_2, log_ll_k_1, log_ll_k_2, log_ll_k, merge_prob
):
    # use log for overflows
    if N_k_1 == 0:
        lgamma_1 = 0
    else:
        lgamma_1 = lgamma(torch.tensor(N_k_1))
    if N_k_2 == 0:
        lgamma_2 = 0
    else:
        lgamma_2 = lgamma(torch.tensor(N_k_2))
    # Hastings ratio in log space
    N_k = N_k_1 + N_k_2
    if N_k > 0:
        H = (
            (lgamma(torch.tensor(N_k)) - (np.log(alpha) + lgamma_1 + lgamma_2))
            + (log_ll_k - (log_ll_k_1 + log_ll_k_2))
        )
    else:
        H = torch.ones(1)

    merge_prob = merge_prob or torch.exp(H)
    return bool(H > 0)

def split_and_merge_op(u_feat, args, index=0, stage=0):
    class_num = args.class_num
    results = {
        'centroids': [],
        'density': [],
        'im2cluster': [],
    }

    u_feat = u_feat.cpu().detach().numpy()

    # 执行高斯混合模型聚类分析
    gmm = GaussianMixture(n_components=class_num, random_state=0)
    gmm.fit(u_feat)

    centroids = gmm.means_

    # 获取每个样本的聚类分配
    labels = gmm.predict(u_feat)

    pred = labels

    labels = torch.tensor(labels)
    centroids = torch.tensor(centroids)
    u_feat = torch.tensor(u_feat)

    prior = Priors(args, class_num, u_feat.shape[1], )
    prior.init_priors(u_feat)

    _, counts = torch.unique(labels, return_counts=True)
    counts = counts.cpu()

    # 所有可能的类的序号范围
    all_classes = torch.arange(class_num)

    # 创建一个布尔掩码来标记哪些类的序号未在 index 中出现
    mask = torch.ones(all_classes.size(0), dtype=torch.bool)
    mask[_] = False

    # 找出缺失的类
    missing_indices = all_classes[mask]

    # 将缺失的类的序号和相应的0样本数量补入
    new_index = torch.cat((_, missing_indices))
    new_value = torch.cat((counts, torch.zeros(missing_indices.size(0), dtype=counts.dtype)))

    # 根据index排序
    sorted_indices = torch.argsort(new_index)
    _ = new_index[sorted_indices]
    counts = new_value[sorted_indices]

    print(_)
    print(counts)

    pi = counts / float(len(u_feat))

    data_covs = compute_data_covs_hard_assignment(labels, u_feat, class_num, centroids.cpu(), prior) # 5,64,64

    # NOTE: the following is to update the mu and cov using a prior. Can be disabled.
    mus = prior.compute_post_mus(counts, centroids.cpu())    # 5,64

    covs = []

    for k in range(len(centroids)):
        feat_k = u_feat[labels == k]
        # print(counts, data_covs.shape)
        cov_k = prior.compute_post_cov(counts[k], feat_k.mean(axis=0), data_covs[k])  # len(codes_k) == counts[k]? yes
        covs.append(cov_k)
    covs = torch.stack(covs)    # 5,64,64

    # NOTE: now we have mus, covs, pi, labels for the global GMM
    sub_clusters, mu_sub_list, covs_sub_list, pi_sub_list, class_sub_assign_list = get_sub_cluster_with_sskmeans(u_feat, labels, prior, args)

    # print(sub_clusters[0]) # [class_label, (mu_sub, cov_sub, pi_sub, class_sub_assign)] * N

    # NOTE: now we have sub_mus, sub_assignments, we can compute split rules now
    labelled_clusters = labels.unique()    # [0, 1, 2, ..., K]


    split_decisions = []
    sub_center = [[] for _ in range(len(labelled_clusters))]

    for _label in labelled_clusters:
        class_indices = labels == _label
        if len(u_feat[class_indices]) <= 5:
            # small clusters will not be splited
            split_decision = False
            split_decisions.append([_label.item(), split_decision])
            continue

        count = 0
        mu_sub_0 = torch.zeros_like(mu_sub_list[0][0])
        mu_sub_1 = torch.zeros_like(mu_sub_list[0][0])

        for i, bool in enumerate(class_indices):
            if bool:
                mu_sub_0 += mu_sub_list[i][0]
                mu_sub_1 += mu_sub_list[i][1]
                _class_sub_assign = class_sub_assign_list[i]
                count += 1

        mu_sub_0 /= count # means_mu_sub_0
        mu_sub_1 /= count # means_mu_sub_0
        num_sub_1 = sum(_class_sub_assign)
        num_sub_0 = len(_class_sub_assign) - num_sub_1
        class_feat = u_feat[class_indices]
        feat_sub_0 = class_feat[_class_sub_assign == 0]  # num_sub_0 * d
        feat_sub_1 = class_feat[_class_sub_assign == 1]  # num_sub_1 * d
        sub_center[_label].append(mu_sub_0)
        sub_center[_label].append(mu_sub_1)

        # print(mus.shape)

        split_decision = split_rule(class_feat, mus[_label], feat_sub_0, feat_sub_1, mu_sub_0, mu_sub_1, num_sub_0, num_sub_1, prior)
        split_decisions.append([_label.item(), split_decision])



    # for class_label, items in tqdm(sub_clusters):
    #     # if class_label in labelled_clusters:
    #     #     # NOTE: labelled clusters are not considered
    #     #     continue
    #     class_indices = labels == class_label
    #     print(class_indices)
    #     assert 0
    #     mu_subs, cov_subs, pi_subs, sub_assign = items
    #     split_decision = split_rule(u_feat[class_indices], sub_assign, prior, mus[class_label], mu_subs)
    #     split_decisions.append([class_label, split_decision])

    print(split_decisions)

    remain_for_merge = np.array([class_l for class_l, split_d in split_decisions if not split_d])
    remain_mus = centroids[remain_for_merge].cpu()
    remain_covs = covs[remain_for_merge]
    remain_pi = pi[remain_for_merge]

    # import ipdb; ipdb.set_trace()
    merge_decisions = []

    mu_nn = NearestNeighbors(n_neighbors=2, metric='euclidean').fit(remain_mus.cpu().detach().numpy())

    for remain_idx, class_label in tqdm(enumerate(remain_for_merge)):
        nn = mu_nn.kneighbors(centroids[class_label].reshape(1, -1).cpu().detach().numpy(), return_distance=False)[0][1:]
        nn = nn.item()
        merge_decision = merge_rule(remain_mus[remain_idx], remain_covs[remain_idx], remain_pi[remain_idx],
                                    remain_mus[nn], remain_covs[nn], remain_pi[nn],
                                    u_feat[labels == class_label], u_feat[labels == remain_for_merge[nn]],
                                    prior)

        Repeated = False
        for _cls, _, _nn in merge_decisions:
            if _cls == nn and _nn == class_label:
                Repeated = True

        if not Repeated:
            merge_decisions.append([class_label, merge_decision, nn])

    print(merge_decisions)

    # NOTE: now we have split_decisions and merge_decisions, we can update the results
    new_centroids = None
    not_updated_idx = labelled_clusters.cpu().numpy().tolist()
    not_updated_idx += [idx for idx, split_d in split_decisions if not split_d]
    not_updated_idx += [idx for idx, merge_d, nn in merge_decisions if not merge_d]
    not_updated_idx = list(set(not_updated_idx))

    print('not_updated_idx:', not_updated_idx)

    new_centroids = centroids[not_updated_idx].cpu()


    # perform split
    for class_label, split_d in split_decisions:
        if split_d:
            mu_subs_0 = sub_center[class_label][0]
            mu_subs_1 = sub_center[class_label][1]
            new_centroids[class_label] = mu_subs_0
            new_centroids = torch.cat((new_centroids, mu_subs_1.unsqueeze(0)), dim=0)

    # print('new centroids:', new_centroids.shape)
    # assert 0


    # perform merge
    indices_to_delete = []

    for class_label, merge_d, nn in merge_decisions:
        if merge_d:
            nn_class_label = remain_for_merge[nn]
            mean_mu = (centroids[class_label] + centroids[nn_class_label]) / 2
            new_centroids = torch.cat((new_centroids, mean_mu.reshape(1, -1).cpu()))
            indices_to_delete.append(class_label)
            indices_to_delete.append(nn)

    centroids = new_centroids
    print('after merge', centroids.shape)

    # 创建一个布尔掩码来选择要保留的向量
    indices_to_delete = list(set(indices_to_delete))
    mask = torch.ones(centroids.size(0), dtype=torch.bool)
    mask[indices_to_delete] = False

    # 使用布尔掩码来选择要保留的向量
    centroids = centroids[mask]

    print('after delete', centroids.shape)
    # assert 0

    dist = pairwise_distance(u_feat.cpu(), centroids.cpu())
    # print(dist.shape)
    # assert 0
    _, pred = torch.min(dist, dim=1)

    # pred[:len(l_targets)] = l_targets

    # # update densities
    # dist2center = torch.sum((u_feat.cpu() - centroids[pred].cpu()) ** 2, dim=1).sqrt()
    #
    # density = torch.zeros(len(centroids))
    # for center_id in pred.unique():
    #     if (pred == center_id).nonzero().shape[0] > 1:
    #         item = dist2center[pred == center_id]
    #         density[center_id.item()] = item.mean() / np.log(len(item) + 10)
    #
    # dmax = density.max()
    # for center_id in pred.unique():
    #     if (pred == center_id).nonzero().shape[0] <= 1:
    #         density[center_id.item()] = dmax
    #
    # density = density.cpu().detach().numpy()
    #
    # density = density.clip(np.percentile(density, 10), np.percentile(density, 90))  # clamp extreme values for stability
    # density = args.temperature * density / density.mean()  # scale the mean to temperature
    # density = torch.from_numpy(density).cuda()

    centroids = F.normalize(centroids, p=2, dim=1).cuda()

    return centroids


from torch import mvlgamma
from torch import lgamma
import numpy as np


class Priors:
    '''
    A prior that will hold the priors for all the parameters.
    '''

    def __init__(self, args, K, codes_dim, counts=10, prior_sigma_scale=None):
        self.name = "prior_class"
        self.pi_prior_type = args.pi_prior  # uniform
        if args.pi_prior:
            self.pi_prior = Dirichlet_prior(K, args.pi_prior, counts)
        else:
            self.pi_prior = None
        self.mus_covs_prior = NIW_prior(args, prior_sigma_scale)

        self.name = self.mus_covs_prior.name
        self.pi_counts = args.prior_dir_counts  # 0.1

    def update_pi_prior(self, K_new, counts=10, pi_prior=None):
        # pi_prior = None- keep the same pi_prior type
        if self.pi_prior:
            if pi_prior:
                self.pi_prioir = Dirichlet_prior(K_new, pi_prior, counts)
            self.pi_prior = Dirichlet_prior(K_new, self.pi_prior_type, counts)

    def comp_post_counts(self, counts):
        if self.pi_prior:
            return self.pi_prior.comp_post_counts(counts)
        else:
            return counts

    def comp_post_pi(self, pi):
        if self.pi_prior:
            return self.pi_prior.comp_post_pi(pi, self.pi_counts)
        else:
            return pi

    def get_sum_counts(self):
        return self.pi_prior.get_sum_counts()

    def init_priors(self, codes):
        return self.mus_covs_prior.init_priors(codes)

    def compute_params_post(self, codes_k, mu_k):
        return self.mus_covs_prior.compute_params_post(codes_k, mu_k)

    def compute_post_mus(self, N_ks, data_mus):
        return self.mus_covs_prior.compute_post_mus(N_ks, data_mus)

    def compute_post_cov(self, N_k, mu_k, data_cov_k):
        return self.mus_covs_prior.compute_post_cov(N_k, mu_k, data_cov_k)

    def log_marginal_likelihood(self, codes_k, mu_k):
        return self.mus_covs_prior.log_marginal_likelihood(codes_k, mu_k)


class Dirichlet_prior:
    def __init__(self, K, pi_prior="uniform", counts=10):
        self.name = "Dirichlet_dist"
        self.K = K
        self.counts = counts
        if pi_prior == "uniform":
            self.p_counts = torch.ones(K) * counts
            self.pi = self.p_counts / float(K * counts)

    def comp_post_counts(self, counts=None):
        if counts is None:
            counts = self.counts
        return counts + self.p_counts

    def comp_post_pi(self, pi, counts=None):
        if counts is None:
            # counts = 0.001
            counts = 0.1
        return (pi + counts) / (pi + counts).sum()

    def get_sum_counts(self):
        return self.K * self.counts


class NIW_prior:
    """A class used to store niw parameters and compute posteriors.
    Used as a class in case we will want to update these parameters.
    """

    def __init__(self, args, prior_sigma_scale=None):
        self.name = "NIW"
        self.prior_mu_0_choice = args.prior_mu_0  # data_mean
        self.prior_sigma_choice = args.prior_sigma_choice  # isotropic
        self.prior_sigma_scale = prior_sigma_scale or args.prior_sigma_scale  # .005
        self.niw_kappa = args.prior_kappa  # 0.0001
        self.niw_nu = args.prior_nu  # at least feat_dim + 1

    def init_priors(self, codes):
        if self.prior_mu_0_choice == "data_mean":
            self.niw_m = codes.mean(axis=0)
        if self.prior_sigma_choice == "isotropic":
            self.niw_psi = (torch.eye(codes.shape[1]) * self.prior_sigma_scale).double()
        elif self.prior_sigma_choice == "data_std":
            self.niw_psi = (torch.diag(codes.std(axis=0)) * self.prior_sigma_scale).double()
        else:
            raise NotImplementedError()
        return self.niw_m, self.niw_psi

    def compute_params_post(self, codes_k, mu_k):
        # This is in HARD assignment.
        N_k = len(codes_k)
        sum_k = codes_k.sum(axis=0)
        kappa_star = self.niw_kappa + N_k
        nu_star = self.niw_nu + N_k
        mu_0_star = (self.niw_m * self.niw_kappa + sum_k) / kappa_star
        codes_minus_mu = codes_k - mu_k
        S = codes_minus_mu.T @ codes_minus_mu
        psi_star = (
                self.niw_psi
                + S
                + (self.niw_kappa * N_k / kappa_star)
                * (mu_k - self.niw_m).unsqueeze(1)
                @ (mu_k - self.niw_m).unsqueeze(0)
        )
        return kappa_star, nu_star, mu_0_star, psi_star

    def compute_post_mus(self, N_ks, data_mus):
        # N_k is the number of points in cluster K for hard assignment, and the sum of all responses to the K-th cluster for soft assignment
        return ((N_ks.reshape(-1, 1) * data_mus) + (self.niw_kappa * self.niw_m)) / (
                N_ks.reshape(-1, 1) + self.niw_kappa
        )

    def compute_post_cov(self, N_k, mu_k, data_cov_k):
        # If it is hard assignments: N_k is the number of points assigned to cluster K, x_mean is their average
        # If it is soft assignments: N_k is the r_k, the sum of responses to the k-th cluster, x_mean is the data average (all the data)
        D = len(mu_k)
        if N_k > 0:
            return (
                    self.niw_psi
                    + data_cov_k * N_k  # unnormalize
                    + (
                            ((self.niw_kappa * N_k) / (self.niw_kappa + N_k))
                            * ((mu_k - self.niw_m).unsqueeze(1) * (mu_k - self.niw_m).unsqueeze(0))
                    )
            ) / (self.niw_nu + N_k + D + 2)
        else:
            return self.niw_psi

    def log_marginal_likelihood(self, codes_k, mu_k):
        kappa_star, nu_star, mu_0_star, psi_star = self.compute_params_post(
            codes_k, mu_k
        )
        (N_k, D) = codes_k.shape
        return (
                -(N_k * D / 2.0) * np.log(np.pi)
                + mvlgamma(torch.tensor(nu_star / 2.0), D)
                - mvlgamma(torch.tensor(self.niw_nu) / 2.0, D)
                + (self.niw_nu / 2.0) * torch.logdet(self.niw_psi)
                - (nu_star / 2.0) * torch.logdet(psi_star)
                + (D / 2.0) * (np.log(self.niw_kappa) - np.log(kappa_star))
        )