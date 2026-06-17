import scanpy as sc
import numpy as np
import os
import re
import json
import argparse
from sklearn.metrics import adjusted_rand_score, silhouette_score, normalized_mutual_info_score
from sklearn.cluster import KMeans
from tqdm import tqdm
import matplotlib.pyplot as plt
from sc_utils import plot_umap


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate and plot embeddings (scFoundation, PCA, scVI) against cell type labels."
    )
    parser.add_argument(
        "--embeddings_path",
        type=str,
        required=True,
        help="Path to the directory containing scFoundation embedding .npy files."
    )
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to the directory containing the raw .h5ad files."
    )
    parser.add_argument(
        "--datasets",
        type=str,
        default=json.dumps({
            'Blood': 'b225ee37-5e06-4e49-9c25-c3d7b5008dab.h5ad',
            'Thymus': 'da951ed6-59c0-4c13-94dc-aff8ff88dc32.h5ad',
            'Lung': '40f8b1a3-9f76-4ac4-8761-32078555ed4e.h5ad',
            'Mammary': '95aa14c9-5226-48ae-bd6c-eb901fb5af7e.h5ad',
            'BMarrow': 'c7f0c3ea-2083-4d87-a8e0-7f69626aa40d.h5ad',
        }),
        help="JSON string mapping dataset names to .h5ad filenames."
    )
    return parser.parse_args()


def main():
    args = parse_args()

    path = args.embeddings_path
    data_path = args.data_path
    datasets = json.loads(args.datasets)

    Emb_files = os.listdir(path)
    Embedding_files = [re.split('_', i)[0] for i in Emb_files]

    scFoundation_scores = {}
    PCA_scores = {}
    scVI_scores = {}

    for name, file in tqdm(datasets.items()):
        adata = sc.read_h5ad(os.path.join(data_path, file))
        print('Loaded single cell data for: ' + name)
        Dataset = 'Tabula sapiens ' + name

        # Read file with embeddings
        file_emb = Emb_files[Embedding_files.index(name)]
        scF_emb = np.load(os.path.join(path, file_emb))
        print('Loaded scFoundation cell embeddings for: ' + name)

        scF_per = plot_umap(scF_emb, adata.obs["cell_type"], title=Dataset, filename=Dataset + '_scFoundation_')
        scFoundation_scores[name] = scF_per

        PCA_per = plot_umap(adata.obsm["X_pca"], adata.obs["cell_type"], title=Dataset, filename=Dataset + '_PCA_')
        PCA_scores[name] = PCA_per

        sCVI_per = plot_umap(adata.obsm["X_scvi"], adata.obs["cell_type"], title=Dataset, filename=Dataset + '_scVI_')
        scVI_scores[name] = sCVI_per

    print(scFoundation_scores)
    print(PCA_scores)
    print(scVI_scores)


if __name__ == "__main__":
    main()
