import scanpy as sc
import numpy as np
import os
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.cluster import KMeans
from tqdm import tqdm
import re
from sc_utils import evaluate_embeddings

import argparse


parser.add_argument('--path', type=str, required=True, help='Path to the h5ad file')

args = parser.parse_args()


path = args.path


datasets = {
    'Blood' : 'b225ee37-5e06-4e49-9c25-c3d7b5008dab.h5ad',
    'Thymus' : 'da951ed6-59c0-4c13-94dc-aff8ff88dc32.h5ad',
    'Lung' : '40f8b1a3-9f76-4ac4-8761-32078555ed4e.h5ad',
    'Mammary' : '95aa14c9-5226-48ae-bd6c-eb901fb5af7e.h5ad',
    'BMarrow' : 'c7f0c3ea-2083-4d87-a8e0-7f69626aa40d.h5ad',
}

Emb_files = os.listdir(path)
Embedding_files = [re.split('_',i)[0] for i in os.listdir(path) ]

scFoundation_scores = {}

for name, file in tqdm(datasets.items()):
    adata = sc.read_h5ad('/home/lmartinez/'+file)
    print('Loaded single cell data for: ' + name )
    # Read file with embeddings
    file_emb = Emb_files[Embedding_files.index(name)]
    scF_emb = np.load(path + '/' + file_emb)
    print('Loaded scFoundation cell embeddings for: ' + name)
    scF_per = evaluate_embeddings(scF_emb, adata.obs.cell_type)
    scFoundation_scores[name] = scF_per

evaluate_embeddings(adata.obsm['X_pca'], adata.obs.cell_type)
