from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
import matplotlib.pyplot as plt
import scanpy as sc

def plot_umap(embedding, labels, title, filename=None, dpi=300):
    if filename is None:
        filename = title.replace(" ", "_").lower() + ".png"
    # Create temp AnnData for visualization
    adata_vis = sc.AnnData(embedding)
    adata_vis.obs["cell_type"] = labels.values.astype(str)
    sc.pp.neighbors(adata_vis, use_rep="X")
    sc.tl.leiden(adata_vis)
    # Evaluate Leiden
    predicted_labels = adata_vis.obs.leiden
    ari_leiden = adjusted_rand_score(labels, predicted_labels)
    nmi_leiden = normalized_mutual_info_score(labels, predicted_labels)
    print(f"  -> ARI Leiden: {ari_leiden:.4f}" )
    print(f"  -> NMI Leiden: {nmi_leiden:.4f}" )
    sc.tl.umap(adata_vis)
    sc.pl.umap(
        adata_vis,
        color="cell_type",
        title=title,
        frameon=False,
        show=False
    )
    plt.savefig(filename, dpi=dpi, bbox_inches="tight")
    plt.close()
    return ari_leiden, nmi_leiden
