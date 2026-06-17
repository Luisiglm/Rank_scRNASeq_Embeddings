
import scanpy as sc
from sklearn.metrics import adjusted_rand_score, silhouette_score
import matplotlib.pyplot as plt
import matplotlib as mpl
import anndata as ad
# Set PDF fonttype to 42 (TrueType)
mpl.rcParams['pdf.fonttype'] = 42

import numpy as np
import pandas as pd
import umap
import scvi
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from tqdm import tqdm
import os

os.makedirs('figures', exist_ok=True)



# Read the h5ad file
adata_tb = sc.read_h5ad('/home/lmartinez/Data/b225ee37-5e06-4e49-9c25-c3d7b5008dab.h5ad')
#Single Cell Sequencing of Human PBMCs in Clonal Hematopoeisis of Indeterminant Potential
adata_other =  sc.read_h5ad('/home/lmartinez/0d245eaa-4c23-4f0b-8ebb-703ec7d87c61.h5ad')

adata_tb.obs["study"] = "Tabula Sapiens Blood"
adata_other.obs["study"] = "Heimlich et al. (2024) Blood Advances"
sc.pp.highly_variable_genes(adata_tb)




d = 1152
max_seq = 4096

def get_cell_embeddings(counts, d, max_seq, gene_emb):
    ranking = np.argsort(-counts, axis=1)
    cell_emb = np.zeros((counts.shape[0], d))
    for i in range(counts.shape[0]):
        mask = counts[i, ranking[i, :]] == 0.0
        if (~mask).any():
            if mask.argmax() > max_seq:
                R_i = ranking[i, :max_seq]
            else:
                R_i = ranking[i, :mask.argmax()]
            cell_emb[i, :] = np.mean(gene_emb[R_i, :], axis=0)
    return cell_emb



gene_emb = np.random.normal(0,1/(d**.5), (20000, d))



rank_emb = get_cell_embeddings(np.asarray(Z)[val_set,:], d, max_seq, gene_emb)

def simulate_scrna(n=1000, batches=2, cell_types=5, b_effect = 0.1,
                   n_housekeeping=200, c_markers=100,
                   n_expressed=2000, genes=20000,
                   m_libsize=4000, sd_libsize=1000):
    """
    Simulate scRNA-seq data using the model:
        log(mu_ij) = X_i @ B + gamma_j + log(N_i) + batch_fx
    where:
        X_i     : (cell_types,) one-hot cell type indicator for cell i
        B       : (cell_types, genes) cell-type effect matrix
        gamma   : (genes,) baseline log-expression
        N_i     : library size for cell i
        batch_fx: (batches, genes) batch effect on log scale
    """
    total_markers = c_markers * cell_types
    n_other = n_expressed - n_housekeeping - c_markers  # per-cell-type private markers
    assert n_housekeeping + total_markers <= genes, \
        "n_housekeeping + c_markers * cell_types exceeds gene budget"
    assert n_other >= 0, \
        "n_expressed too small to fit housekeeping + markers"
    assert n_expressed <= genes, \
        "n_expressed cannot exceed total genes"
    # Initialise B, phi, gamma — all silent by default
    B     = np.zeros((cell_types, genes))
    phi   = np.zeros(genes)
    gamma = np.full(genes, -20.0)   # exp(-20) ≈ 0; structural zeros
    # Housekeeping genes [0 : n_housekeeping]
    #    Expressed in all cell types, low overdispersion
    hk = np.arange(n_housekeeping)
    B[:, hk]   = np.random.normal(0.1, 0.05, (cell_types, n_housekeeping))
    phi[hk]    = 1/np.random.uniform(20, 50, n_housekeeping)
    gamma[hk]  = np.random.normal(-2, 0.5, n_housekeeping)
    # Marker genes — c_markers exclusive genes per cell type
    #    Strong signal, moderate overdispersion
    genes_list   = list(range(n_housekeeping, genes))
    marker_indices = []
    for i in range(cell_types):
        idx = np.random.choice(genes_list, c_markers, replace=False)
        marker_indices.append(idx)
        B[i, idx]  = np.random.normal(4.0, 0.2, c_markers)
        phi[idx]   = 1/np.random.uniform(0.5, 2.0, c_markers)
        gamma[idx] = np.random.normal(-4, 0.5, c_markers)
        genes_list = list(set(genes_list) - set(idx))
    #  Other expressed genes — one shared pool, each cell type
    #    activates a random 60% subset, low effect size, higher r
    other_idx = np.array(np.random.choice(genes_list, n_other, replace=False))
    phi[other_idx]   = 1/np.random.uniform(2, 10, n_other)
    gamma[other_idx] = np.random.normal(-0.5, 0.5, n_other)
    for i in range(cell_types):
        active = np.random.choice(other_idx, int(n_other * 0.6), replace=False)
        B[i, active] = np.random.normal(0.05, 0.05, len(active))
    # Cell assignments and library sizes
    cell_type_labels = np.random.choice(cell_types, n)
    batch_labels     = np.random.choice(batches, n)
    # Log-normal library sizes — always positive, CV ≈ sd/mean
    log_N = np.random.normal(np.log(m_libsize), sd_libsize / m_libsize, n)
    N     = np.exp(log_N)
    # Batch effects: zero-mean log-scale per gene per batch
    expressed_mask = phi > 0
    batch_fx = np.zeros((batches, genes))
    batch_fx[:, expressed_mask] = np.random.normal(
        0, b_effect, (batches, expressed_mask.sum())
    )
    # Compute mu and sample counts
    expressed_mask = phi > 0
    X = np.zeros((n, cell_types))
    X[np.arange(n), cell_type_labels] = 1
    # log(mu) = X @ B + gamma + log(N) + batch_fx   shape: (n, genes)
    log_mu = (X @ B) + gamma[None, :] + batch_fx[batch_labels, :]
    mu = np.exp(log_mu)
    mu[:, ~expressed_mask] = 0.0
    mu = mu / mu.sum(axis=1, keepdims=True) * N[:, None]
    counts = np.zeros((n, genes), dtype=np.int32)
    for i in range(n):
        counts[i] = sample_counts(mu[i], phi)
    return {
        'counts':           counts,           # (n, genes)
        'cell_type_labels': cell_type_labels, # (n,)
        'batch_labels':     batch_labels,     # (n,)
        'B':                B,                # (cell_types, genes)
        'gamma':            gamma,            # (genes,)
        'phi':              phi,              # (genes,)
        'log_N':            log_N,            # (n,)
        'marker_indices':   marker_indices,   # list of (cell_types,) arrays
        'mu':               mu,
    }




B_range = [  0.0, 0.5, 1.0, 4.0]
fig, ax = plt.subplots(2,4, figsize=(12,6), constrained_layout=True)
asw_pca = []
asw_ranking = []
for i in tqdm(range(len(B_range))):
    results =  simulate_scrna(b_effect = B_range[i])
    raw_counts = results['counts']
    obs = pd.DataFrame({
        'batch': [f'batch_{b}' for b in results['batch_labels']],
        'cell_type' : [f'cell_{b}' for b in results['cell_type_labels']],
    })
    sim_adata = ad.AnnData(X = raw_counts, obs = obs )
    sc.pp.normalize_total(sim_adata, target_sum=1e4)
    sc.pp.log1p(sim_adata)
    sc.pp.highly_variable_genes(sim_adata)
    sim_adata = sim_adata[:, sim_adata.var.highly_variable]  # subset to HVGs
    sc.pp.scale(sim_adata)  # zero mean, unit variance
    sc.pp.pca(sim_adata)
    sc.pp.neighbors(sim_adata)
    sc.tl.umap(sim_adata)
    sc.pl.umap(sim_adata, color='cell_type', save = str(B_range[i]) + '_simulated_umap.png')
    asw_pca.append(silhouette_score(sim_adata.obsm['X_pca'], sim_adata.obs.batch, metric='euclidean'))
    rank_emb = get_cell_embeddings(raw_counts, d, max_seq, gene_emb)
    asw_ranking.append(silhouette_score(rank_emb, sim_adata.obs.batch, metric='euclidean'))
    sim_adata.obsm['rank_emb'] = rank_emb
    sc.pp.neighbors(sim_adata, use_rep='rank_emb')
    sc.tl.umap(sim_adata)
    sc.pl.umap(sim_adata, color='cell_type', save=str(B_range[i]) + 'rank_simulated_umap.png')
    print(str(asw_ranking[i]) + ' ' + str(B_range[i]))





# ------------------------------------------------------------------ #
# Results storage
# ------------------------------------------------------------------ #
results_records = []

for i in tqdm(range(len(B_range))):
    for seed in range(n_replicates):
        np.random.seed(seed)
        results = simulate_scrna(b_effect=B_range[i])
        raw_counts = results['counts']
        obs = pd.DataFrame({
            'batch':     [f'batch_{b}' for b in results['batch_labels']],
            'cell_type': [f'cell_{b}' for b in results['cell_type_labels']],
        })
        # ------------------------------------------------------------------ #
        # PCA baseline
        # ------------------------------------------------------------------ #
        sim_adata = ad.AnnData(X=raw_counts, obs=obs)
        sc.pp.normalize_total(sim_adata, target_sum=1e4)
        sc.pp.log1p(sim_adata)
        sc.pp.highly_variable_genes(sim_adata)
        sim_adata = sim_adata[:, sim_adata.var.highly_variable]
        sc.pp.scale(sim_adata)
        sc.pp.pca(sim_adata)
        sc.pp.neighbors(sim_adata)
        sc.tl.umap(sim_adata)
        if seed == 0:
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            sc.pl.umap(sim_adata, color='cell_type', ax=axes[0], show=False,
                       title=f'Cell type (b={B_range[i]:.2f})')
            sc.pl.umap(sim_adata, color='batch',     ax=axes[1], show=False,
                       title=f'Batch (b={B_range[i]:.2f})')
            plt.tight_layout()
            plt.savefig(f'figures/{B_range[i]}_pca_umap.png', dpi=150, bbox_inches='tight')
            plt.close()
        asw_pca_ct    = silhouette_score(sim_adata.obsm['X_pca'], sim_adata.obs['cell_type'], metric='euclidean')
        asw_pca_batch = silhouette_score(sim_adata.obsm['X_pca'], sim_adata.obs['batch'],     metric='euclidean')
        # ------------------------------------------------------------------ #
        # Ranking embedding
        # ------------------------------------------------------------------ #
        rank_emb = get_cell_embeddings(raw_counts, d, max_seq, gene_emb)
        sim_adata.obsm['rank_emb'] = rank_emb
        sc.pp.neighbors(sim_adata, use_rep='rank_emb')
        sc.tl.umap(sim_adata)
        if seed == 0:
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            sc.pl.umap(sim_adata, color='cell_type', ax=axes[0], show=False,
                       title=f'Cell type (b={B_range[i]:.2f})')
            sc.pl.umap(sim_adata, color='batch',     ax=axes[1], show=False,
                       title=f'Batch (b={B_range[i]:.2f})')
            plt.tight_layout()
            plt.savefig(f'figures/{B_range[i]}_rank_umap.png', dpi=150, bbox_inches='tight')
            plt.close()
        asw_rank_ct    = silhouette_score(rank_emb, sim_adata.obs['cell_type'], metric='euclidean')
        asw_rank_batch = silhouette_score(rank_emb, sim_adata.obs['batch'],     metric='euclidean')
        # ------------------------------------------------------------------ #
        # Harmony
        # ------------------------------------------------------------------ #
        sim_adata_harmony = sim_adata.copy()
        sc.pp.neighbors(sim_adata_harmony, use_rep='X_pca')
        sc.external.pp.harmony_integrate(sim_adata_harmony, key='batch')
        sc.pp.neighbors(sim_adata_harmony, use_rep='X_pca_harmony')
        sc.tl.umap(sim_adata_harmony)
        if seed == 0:
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            sc.pl.umap(sim_adata_harmony, color='cell_type', ax=axes[0], show=False,
                       title=f'Cell type (b={B_range[i]:.2f})')
            sc.pl.umap(sim_adata_harmony, color='batch',     ax=axes[1], show=False,
                       title=f'Batch (b={B_range[i]:.2f})')
            plt.tight_layout()
            plt.savefig(f'figures/{B_range[i]}_harmony_umap.png', dpi=150, bbox_inches='tight')
            plt.close()
        asw_harmony_ct    = silhouette_score(sim_adata_harmony.obsm['X_pca_harmony'], sim_adata_harmony.obs['cell_type'], metric='euclidean')
        asw_harmony_batch = silhouette_score(sim_adata_harmony.obsm['X_pca_harmony'], sim_adata_harmony.obs['batch'],     metric='euclidean')
        print(f"b={B_range[i]:.2f} seed={seed} | "
              f"PCA ct={asw_pca_ct:.3f} batch={asw_pca_batch:.3f} | "
              f"Rank ct={asw_rank_ct:.3f} batch={asw_rank_batch:.3f} | "
              f"Harmony ct={asw_harmony_ct:.3f} batch={asw_harmony_batch:.3f} | "


# ------------------------------------------------------------------ #
# Aggregate results
# ------------------------------------------------------------------ #
df = pd.DataFrame(results_records)
df.to_csv('asw_results.csv', index=False)

summary = df.groupby('b_effect').agg(['mean', 'std']).round(3)
print(summary)

# ------------------------------------------------------------------ #
# Summary plot
# ------------------------------------------------------------------ #
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
methods = [('pca', 'PCA'), ('rank', 'Ranking'), ('harmony', 'Harmony'), ('scvi', 'scVI')]
colors  = ['steelblue', 'darkorange', 'green', 'red']

for ax, metric, title in zip(axes, ['ct', 'batch'], ['Cell Type ASW (↑)', 'Batch ASW (↓)']):
    for (key, label), color in zip(methods, colors):
        col   = f'{key}_{metric}'
        mean  = df.groupby('b_effect')[col].mean()
        std   = df.groupby('b_effect')[col].std()
        ax.plot(mean.index, mean.values, label=label, color=color, marker='o')
        ax.fill_between(mean.index, mean - std, mean + std, alpha=0.2, color=color)
    ax.set_xlabel('b_effect')
    ax.set_ylabel('ASW')
    ax.set_title(title)
    ax.legend()

plt.tight_layout()
plt.savefig('figures/asw_summary.png', dpi=150, bbox_inches='tight')
plt.close()





rank_emb = get_cell_embeddings(raw_counts, d, max_seq, gene_emb)
# Save the figure (pass filename positionally)
plt.savefig('Figures_Exp_Rank_Batch_Lasso', dpi=600, bbox_inches="tight")
# Close the figure to free up memory
plt.close(fig)

fig, ax = plt.subplots()
ax.plot(batch_to_signal, asw_ranking, color = 'tab:blue', linewidth = 3 , label = 'Ranked Embedding')
ax.plot(batch_to_signal, asw_pca, color = 'tab:orange', linewidth = 3 , label = 'PCA')
ax.tick_params(axis='both', which='major', labelsize=25)

ax.legend(fontsize = 25)

# Save the figure (pass filename positionally)
plt.savefig('Figures_Sillhouette', dpi=600, bbox_inches="tight")
# Close the figure to free up memory
plt.close(fig)
