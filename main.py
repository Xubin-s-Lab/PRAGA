import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
import torch
import pandas as pd
import scanpy as sc
import numpy as np
import argparse
import time
from PRAGA.preprocess import fix_seed
from PRAGA.preprocess import clr_normalize_each_cell, pca
from PRAGA.preprocess import construct_neighbor_graph, lsi
from PRAGA.Train_model import Train
from PRAGA.Train_model_3M import Train_3M
from PRAGA.utils import clustering
from PRAGA import preprocess, preprocess_3M

def main(args):
    # define device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # read data
    if args.data_type in ['10x', 'SPOTS', 'Stereo-CITE-seq']:
        adata_omics1 = sc.read_h5ad(os.path.join(args.file_fold, 'adata_RNA.h5ad'))
        adata_omics2 = sc.read_h5ad(os.path.join(args.file_fold, 'adata_ADT.h5ad'))
    elif args.data_type == 'Spatial-epigenome-transcriptome':
        adata_omics1 = sc.read_h5ad(os.path.join(args.file_fold, 'adata_RNA.h5ad'))
        adata_omics2 = sc.read_h5ad(os.path.join(args.file_fold, 'adata_peaks_normalized.h5ad'))
    elif args.data_type == 'Simulation':
        adata_omics1 = sc.read_h5ad(os.path.join(args.file_fold, 'adata_RNA.h5ad'))
        adata_omics2 = sc.read_h5ad(os.path.join(args.file_fold, 'adata_ADT.h5ad'))
        adata_omics3 = sc.read_h5ad(os.path.join(args.file_fold, 'adata_ATAC.h5ad'))

    adata_omics1.var_names_make_unique()
    adata_omics2.var_names_make_unique()
    if args.data_type == 'Simulation':
        adata_omics3.var_names_make_unique()

        # Fix random seed

    random_seed = 2024
    fix_seed(random_seed)

    # Preprocess
    if args.data_type == '10x':
        # RNA
        sc.pp.filter_genes(adata_omics1, min_cells=10)
        sc.pp.highly_variable_genes(adata_omics1, flavor="seurat_v3", n_top_genes=3000)
        sc.pp.normalize_total(adata_omics1, target_sum=1e4)
        sc.pp.log1p(adata_omics1)
        sc.pp.scale(adata_omics1)
        adata_omics1_high = adata_omics1[:, adata_omics1.var['highly_variable']]
        adata_omics1.obsm['feat'] = pca(adata_omics1_high, n_comps=adata_omics2.n_vars - 1)
        # Protein
        adata_omics2 = clr_normalize_each_cell(adata_omics2)
        sc.pp.scale(adata_omics2)
        adata_omics2.obsm['feat'] = pca(adata_omics2, n_comps=adata_omics2.n_vars - 1)
        data = preprocess.construct_neighbor_graph(adata_omics1, adata_omics2, datatype=args.data_type, Arg=args)
    elif args.data_type == 'Spatial-epigenome-transcriptome':
        # RNA
        sc.pp.filter_genes(adata_omics1, min_cells=10)
        sc.pp.filter_cells(adata_omics1, min_genes=200)
        sc.pp.highly_variable_genes(adata_omics1, flavor="seurat_v3", n_top_genes=3000)
        sc.pp.normalize_total(adata_omics1, target_sum=1e4)
        sc.pp.log1p(adata_omics1)
        sc.pp.scale(adata_omics1)
        adata_omics1_high = adata_omics1[:, adata_omics1.var['highly_variable']]
        adata_omics1.obsm['feat'] = pca(adata_omics1_high, n_comps=50)
        # ATAC
        adata_omics2 = adata_omics2[
            adata_omics1.obs_names].copy()  # .obsm['X_lsi'] represents the dimension reduced feature
        if 'X_lsi' not in adata_omics2.obsm.keys():
            sc.pp.highly_variable_genes(adata_omics2, flavor="seurat_v3", n_top_genes=3000)
            lsi(adata_omics2, use_highly_variable=False, n_components=51)
        adata_omics2.obsm['feat'] = adata_omics2.obsm['X_lsi'].copy()
        data = preprocess.construct_neighbor_graph(adata_omics1, adata_omics2, datatype=args.data_type, Arg=args)
    elif args.data_type == 'SPOTS':
        # RNA
        sc.pp.filter_genes(adata_omics1, min_cells=10)
        sc.pp.highly_variable_genes(adata_omics1, flavor="seurat_v3", n_top_genes=3000)
        sc.pp.normalize_total(adata_omics1, target_sum=1e4)
        sc.pp.log1p(adata_omics1)
        sc.pp.scale(adata_omics1)
        adata_omics1_high = adata_omics1[:, adata_omics1.var['highly_variable']]
        adata_omics1.obsm['feat'] = pca(adata_omics1_high, n_comps=adata_omics2.n_vars - 1)
        # Protein
        adata_omics2 = clr_normalize_each_cell(adata_omics2)
        sc.pp.scale(adata_omics2)
        adata_omics2.obsm['feat'] = pca(adata_omics2, n_comps=adata_omics2.n_vars - 1)
        data = preprocess.construct_neighbor_graph(adata_omics1, adata_omics2, datatype=args.data_type, Arg=args)
    elif args.data_type == 'Stereo-CITE-seq':
        # RNA
        sc.pp.filter_genes(adata_omics1, min_cells=10)
        sc.pp.filter_cells(adata_omics1, min_genes=80)
        sc.pp.filter_genes(adata_omics2, min_cells=50)
        adata_omics2 = adata_omics2[adata_omics1.obs_names].copy()
        sc.pp.highly_variable_genes(adata_omics1, flavor="seurat_v3", n_top_genes=3000)
        sc.pp.normalize_total(adata_omics1, target_sum=1e4)
        sc.pp.log1p(adata_omics1)
        adata_omics1_high = adata_omics1[:, adata_omics1.var['highly_variable']]
        adata_omics1.obsm['feat'] = pca(adata_omics1_high, n_comps=adata_omics2.n_vars - 1)
        # Protein
        adata_omics2 = clr_normalize_each_cell(adata_omics2)
        adata_omics2.obsm['feat'] = pca(adata_omics2, n_comps=adata_omics2.n_vars - 1)
        data = preprocess.construct_neighbor_graph(adata_omics1, adata_omics2, datatype=args.data_type, Arg=args)
    elif args.data_type == 'Simulation':
        n_protein = adata_omics2.n_vars
        sc.pp.highly_variable_genes(adata_omics1, flavor="seurat_v3", n_top_genes=3000)
        sc.pp.normalize_total(adata_omics1, target_sum=1e4)
        sc.pp.log1p(adata_omics1)
        adata_omics1_high = adata_omics1[:, adata_omics1.var['highly_variable']]
        adata_omics1.obsm['feat'] = pca(adata_omics1_high, n_comps=n_protein)
        # Protein
        adata_omics2 = clr_normalize_each_cell(adata_omics2)
        adata_omics2.obsm['feat'] = pca(adata_omics2, n_comps=n_protein)
        # ATAC
        sc.pp.highly_variable_genes(adata_omics3, flavor="seurat_v3", n_top_genes=3000)
        lsi(adata_omics3, use_highly_variable=False, n_components=n_protein + 1)
        adata_omics3.obsm['feat'] = adata_omics3.obsm['X_lsi'].copy()

        data = preprocess_3M.construct_neighbor_graph(adata_omics1, adata_omics2, adata_omics3)
    else:
        assert 0

    # define model
    if args.data_type == 'Simulation':
        model = Train_3M(data, datatype=args.data_type, device=device, Arg=args)
    else:
        model = Train(data, datatype=args.data_type, device=device, Arg=args)

    start_time = time.time()

    # train model
    output = model.train()

    end_time = time.time()

    print("Training time: ", end_time - start_time)
    # torch.save(model.model.state_dict(), 'model_weights/PRAGA_' + args.data_type + '.pth')

    adata = adata_omics1.copy()
    adata.obsm['emb_latent_omics1'] = output['emb_latent_omics1'].copy()
    adata.obsm['emb_latent_omics2'] = output['emb_latent_omics2'].copy()
    adata.obsm['PRAGA'] = output['PRAGA'].copy()

    # Clustering
    tool = 'mclust' # mclust, leiden, and louvain
    clustering(adata, key='PRAGA', add_key='PRAGA', n_clusters=args.n_clusters, method=tool, use_pca=True)

    label = adata.obs['PRAGA']

    if args.data_type == 'Simulation':
        ids = label.index.astype(str).str[:4]
        int_list = [int(num_str) for num_str in ids]
        label_list = [-1 for i in range(len(int_list))]
        for i in range(len(int_list)):
            label_list[int_list[i]] = label[i]
        spot_size = 60
    else:
        label_list = label.tolist()
        spot_size = 20

    # Save results
    output_file = args.txt_out_path
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        for num in label_list:
            f.write(f"{num}\n")

    # visualization

    if args.data_type == 'Stereo-CITE-seq':
        adata.obsm['spatial'][:, 1] = -1 * adata.obsm['spatial'][:, 1]
    elif args.data_type == 'SPOTS':
        # flip tissue image
        import numpy as np
        adata.obsm['spatial'] = np.rot90(np.rot90(np.rot90(np.array(adata.obsm['spatial'])).T).T).T
        adata.obsm['spatial'][:, 1] = -1 * adata.obsm['spatial'][:, 1]

    import matplotlib.pyplot as plt
    fig, ax_list = plt.subplots(1, 2, figsize=(7, 3))
    sc.pp.neighbors(adata, use_rep='PRAGA', n_neighbors=500)
    sc.tl.umap(adata)

    sc.pl.umap(adata, color='PRAGA', ax=ax_list[0], title='PRAGA', s=spot_size, show=False)
    sc.pl.embedding(adata, basis='spatial', color='PRAGA', ax=ax_list[1], title='PRAGA', s=spot_size, show=False)

    plt.tight_layout(w_pad=0.3)
    plt.savefig(args.vis_out_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Script to modify global variable')
    parser.add_argument('--file_fold', type=str,
                        help='Path to data folder')
    parser.add_argument('--data_type', type=str,
                        choices=['10x', 'Spatial-epigenome-transcriptome', 'SPOTS', 'Stereo-CITE-seq', 'Simulation'],
                        help='data_type')
    parser.add_argument('--n_clusters', type=int,
                        help='n_clusters for clustering')

    parser.add_argument('--init_k', type=int, default=10, help='init k')
    parser.add_argument('--KNN_k', type=int, default=20, help='KNN_k')
    parser.add_argument('--alpha', type=float, default=0.9, help='init k')
    parser.add_argument('--cl_weight', type=float, default=1, help='weight')
    parser.add_argument('--RNA_weight', type=float, default=5, help='weight')
    parser.add_argument('--ADT_weight', type=float, default=5, help='weight')
    parser.add_argument('--tau', type=float, default=2, help='temperature')
    parser.add_argument('--vis_out_path', type=str, default='results/HLN.png', help='vis_out_path')
    parser.add_argument('--txt_out_path', type=str, default='results/HLN.txt', help='txt_out_path')
    args = parser.parse_args()
    main(args)
