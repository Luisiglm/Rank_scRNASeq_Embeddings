import scgpt as scg
from sklearn.cluster import KMeans
import scanpy as sc
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sc_utils import evaluate_embeddings
import json

# Read the h5ad file
Blood = 'b225ee37-5e06-4e49-9c25-c3d7b5008dab.h5ad'
Thymus = 'da951ed6-59c0-4c13-94dc-aff8ff88dc32.h5ad
Lung = '40f8b1a3-9f76-4ac4-8761-32078555ed4e.h5ad'
Mammary = '95aa14c9-5226-48ae-bd6c-eb901fb5af7e.h5ad'
BMarrow = 'c7f0c3ea-2083-4d87-a8e0-7f69626aa40d.h5ad'


adata = sc.read_h5ad('/home/lmartinez/0d245eaa-4c23-4f0b-8ebb-703ec7d87c61.h5ad')


raw_title = adata.uns.get('title', 'Unknown_Dataset') # Default if title missing
# Replace spaces and slashes to make it safe for filenames
Dataset = str(raw_title).replace(" ", "_").replace("/", "-")

# scGPT Embeddings

ref_embed_adata = scg.tasks.embed_data(
    adata,
    '/home/lmartinez/scGPT/',
    gene_col="feature_name",
    batch_size=64,
)



scGPT = evaluate_embeddings(ref_embed_adata.obsm['X_scGPT'], adata.obs.cell_type)

# Save to Json
res_dict = {}
res_dict['Dataset'] = Dataset  # Store dataset name in results
res_dict['ARI'] = dict()
res_dict['Sillhouette'] = dict()

res_dict['ARI']['scGPT'] = scGPt[0]
res_dict['Sillhouette']['scGPT'] = scGPt[1]

output_filename = f"{Dataset}_Geneformer_results.json"
with open(output_filename, 'w') as f:
    json.dump(res_dict, f, indent=4)

print(f"Results saved to {output_filename}")
