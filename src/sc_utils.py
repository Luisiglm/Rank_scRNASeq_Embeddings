from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import scanpy as sc


def evaluate_embeddings(embeddings, labels):
    # Set number of clusters
    n_classes = len(set(labels))
    # K-Means
    kmeans = KMeans(n_clusters=n_classes, random_state=42, n_init=10)
    predicted_labels = kmeans.fit_predict(embeddings)
    # ARI and Sillhouette
    ari = adjusted_rand_score(labels, predicted_labels)
    asw = silhouette_score(embeddings, labels)
    print(f"  -> ARI: {ari:.4f}")
    print(f"  -> ASW: {asw:.4f}")
    return ari, asw


def plot_umap(embedding, labels, title, filename=None, dpi=300):
    if filename is None:
        filename = title.replace(" ", "_").lower() + ".png"
    # Create temp AnnData for visualization
    adata_vis = sc.AnnData(embedding)
    adata_vis.obs["cell_type"] = labels.values.astype(str)
    sc.pp.neighbors(adata_vis, use_rep="X")
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
