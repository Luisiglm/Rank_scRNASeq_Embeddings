import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
import scanpy as sc
from sklearn.metrics import adjusted_rand_score, silhouette_score, normalized_mutual_info_score
from sklearn.cluster import KMeans
from tqdm import tqdm
from sc_utils import plot_umap


parser = argparse.ArgumentParser(description="Random Embeddings Analysis")

parser.add_argument('--address', type=str, required=True, help='Path to the h5ad file')
parser.add_argument('--d_GF', type=int, required=False,   help='Dimension of embedding')
parser.add_argument('--seed', type=int, required=False,   help='Random seed')

args = parser.parse_args()


address = args.address
d_GF = args.d_GF
seed = args.seed

print(f"Loading file from: {address}")


# Read the h5ad file using the provided address
adata = sc.read_h5ad(address)

raw_title = adata.uns.get('title', 'Unknown_Dataset') # Default if title missing
# Replace spaces and slashes to make it safe for filenames
Dataset = str(raw_title).replace(" ", "_").replace("/", "-")


x = adata.X.todense()
ranking = np.argsort(-x, axis=1)

rng = np.random.default_rng(seed)

gene_emb_GF = rng.normal(0, 1/(d_GF)**.5, size=(x.shape[1], d_GF)) 
cell_emb_GF = np.zeros((x.shape[0], d_GF))
cell_emb_2000 = np.zeros((x.shape[0], d_GF))
cell_emb_1000 = np.zeros((x.shape[0], d_GF))
cell_emb_500 = np.zeros((x.shape[0], d_GF))
cell_emb_250 = np.zeros((x.shape[0], d_GF))
cell_emb_125 = np.zeros((x.shape[0], d_GF))

# calculate most variable genes
sc.pp.highly_variable_genes(adata)

hvgs = set(np.where(adata.var['highly_variable'])[0])

Top_4096 = set()
Top_2000 = set()
Top_1000 = set()
Top_500 = set()
Top_250 = set()
Top_125 = set()


for i in range(x.shape[0]):
    mask = x[i, ranking[i, :]] == 0.0
    # Determine the limit for ranking indices
    limit = 4096 if mask.argmax() > 4096 else mask.argmax()
    R_i = ranking[i, :limit]
    Top_4096.update(set(R_i.A1))
    if limit>2000:
        R_2000 = ranking[i, :2000]
    else:
        R_2000 = ranking[i, :limit]
    if limit> 1000:
        R_1000 = ranking[i, :1000]
    else:
        R_1000 = ranking[i, :limit]
    if limit > 500:
        R_500 = ranking[i, :500]
    else:
        R_500 = ranking[i, :limit]
    if limit > 250:
        R_250 = ranking[i, :250]
    else:
        R_250 = ranking[i, :limit]
    if limit > 125:
        R_125 = ranking[i, :125]
    else:
        R_125 = ranking[i, :limit]
    Top_2000.update(set(R_2000.A1))
    Top_1000.update(set(R_1000.A1))
    Top_500.update(set(R_500.A1))
    Top_250.update(set(R_250.A1))
    Top_125.update(set(R_125.A1))
    cell_emb_GF[i, :] = np.mean(gene_emb_GF[R_i, :], axis=1)
    cell_emb_2000[i, :] = np.mean(gene_emb_GF[R_2000, :], axis=1)
    cell_emb_1000[i, :] = np.mean(gene_emb_GF[R_1000, :], axis=1)
    cell_emb_500[i, :] = np.mean(gene_emb_GF[R_500, :], axis=1)
    cell_emb_250[i, :] = np.mean(gene_emb_GF[R_250, :], axis=1)
    cell_emb_125[i, :] = np.mean(gene_emb_GF[R_125, :], axis=1)


res_dict = dict()
from scipy.stats import fisher_exact

# Evaluate intercept!

def evaluate_hgvs(hvgs,Top_k):
    Int = Top_k.intersection(hvgs)
    In_k_and_hgvs = len(Int)
    In_k_not_hgvs = len(Top_k) - In_k_and_hgvs
    In_hgvs_not_k = len(hvgs) - In_k_and_hgvs
    table = [[In_k_and_hgvs , In_k_not_hgvs],[In_hgvs_not_k, len(adata.var)]]
    res = fisher_exact(table, alternative='two-sided')
    return In_k_and_hgvs, res.pvalue

Int_4096, p_4096 = evaluate_hgvs(hvgs,Top_4096)
Int_2000, p_2000 = evaluate_hgvs(hvgs,Top_2000)
Int_1000, p_1000 = evaluate_hgvs(hvgs,Top_1000)
Int_500, p_500 = evaluate_hgvs(hvgs,Top_500)
Int_250, p_250 = evaluate_hgvs(hvgs,Top_250)
Int_125, p_125 = evaluate_hgvs(hvgs,Top_125)



ari_leiden_4096, nmi_leiden_4096 = plot_umap(cell_emb_GF, adata.obs["cell_type"], title = Dataset, filename = Dataset+'_Top_4096_' + str(d_GF))
ari_leiden_2000, nmi_leiden_2000 = plot_umap(cell_emb_2000, adata.obs["cell_type"],title = Dataset, filename =Dataset +'_Top_2000_' +str( d_GF))
ari_leiden_1000, nmi_leiden_1000 = plot_umap(cell_emb_1000, adata.obs["cell_type"], title = Dataset,filename =Dataset+ '_Top_1000_' + str(d_GF))
ari_leiden_500, nmi_leiden_500 = plot_umap(cell_emb_500, adata.obs["cell_type"],title = Dataset, filename = Dataset+'_Top_500_' + str(d_GF))
ari_leiden_250, nmi_leiden_250 = plot_umap(cell_emb_250, adata.obs["cell_type"],title = Dataset, filename =Dataset +'_Top_250_' + str(d_GF))
ari_leiden_125, nmi_leiden_125 = plot_umap(cell_emb_125, adata.obs["cell_type"], title = Dataset,filename =Dataset +'_Top_125_' + str(d_GF))

# write results
results = {}
results['4096'] = {'ARI':ari_leiden_4096, 'NMI':nmi_leiden_4096, 'K_HGVs':Int_4096, 'K Unique': len(Top_4096), 'p_value':p_4096}
results['2000'] = {'ARI':ari_leiden_2000, 'NMI':nmi_leiden_2000, 'K_HGVs':Int_2000, 'K Unique': len(Top_2000), 'p_value':p_2000}
results['1000'] = {'ARI':ari_leiden_1000, 'NMI':nmi_leiden_1000, 'K_HGVs':Int_1000, 'K Unique': len(Top_1000), 'p_value':p_1000}
results['500'] = {'ARI':ari_leiden_500, 'NMI':nmi_leiden_500, 'K_HGVs':Int_500, 'K Unique': len(Top_500), 'p_value':p_500}
results['250'] = {'ARI':ari_leiden_250, 'NMI':nmi_leiden_250, 'K_HGVs':Int_250, 'K Unique': len(Top_250), 'p_value':p_250}
results['125'] = {'ARI':ari_leiden_125, 'NMI':nmi_leiden_125, 'K_HGVs':Int_125, 'K Unique': len(Top_125), 'p_value':p_125}

# Save to file
output_filename = f"{Dataset}_{d_GF}_results_{seed}.json"
with open(output_filename, 'w') as f:
    json.dump(results, f, indent=4)

print(f"Results saved to {output_filename}")


