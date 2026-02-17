import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
import scanpy as sc
import torch
from sklearn.metrics import adjusted_rand_score, silhouette_score
from transformers import AutoModelForMaskedLM
from geneformer import TranscriptomeTokenizer
from sklearn.cluster import KMeans
from tqdm import tqdm

# --- 1. Argument Parsing ---
parser = argparse.ArgumentParser(description="Run Geneformer Analysis")

parser.add_argument('--address', type=str, required=True, help='Path to the h5ad file')

args = parser.parse_args()


address = args.address


print(f"Loading file from: {address}")

# Read the h5ad file using the provided address
adata = sc.read_h5ad(address)


raw_title = adata.uns.get('title', 'Unknown_Dataset') # Default if title missing
# Replace spaces and slashes to make it safe for filenames
Dataset = str(raw_title).replace(" ", "_").replace("/", "-")

# load tokenizer
tokenizer = TranscriptomeTokenizer()

# adata.write(address)

model = AutoModelForMaskedLM.from_pretrained("ctheodoris/Geneformer")
model.eval()


loss_fn = torch.nn.CrossEntropyLoss(reduce=None, reduction='none')
device = 'cuda:1' if torch.cuda.is_available() else 'cpu'

model = model.to(device)
loss_fn = loss_fn.to(device)

gene_ids = adata.var_names.tolist()

# Initialize arrays for embeddings
gf_embeddings_zero = np.zeros((x.shape[0], 1152))
gf_embeddings_full = np.zeros((x.shape[0], 1152))
gf_embeddings_short_30 = np.zeros((x.shape[0], 1152))
gf_embeddings_short_60 = np.zeros((x.shape[0], 1152))
gf_embeddings_short_125 = np.zeros((x.shape[0], 1152))
gf_embeddings_short_250 = np.zeros((x.shape[0], 1152))
gf_embeddings_short_500 = np.zeros((x.shape[0], 1152))

positional_losses = np.zeros((4096,))
counter = np.zeros((4096,))

for i in tqdm(range(x.shape[0])):
    mask = x[i, ranking[i, :]] == 0.0
    R_i = np.array(ranking[i, :mask.argmax()]).flatten()
    sorted_genes = [gene_ids[idx] for idx in R_i]

    input_ids = [tokenizer.gene_token_dict.get('<cls>')] + \
                [tokenizer.gene_token_dict.get(gene) for gene in sorted_genes]
    input_ids += [tokenizer.gene_token_dict.get('<eos>')]

    input_ids = [val for val in input_ids if val is not None]

    if len(input_ids) > 4096:
        input_ids = input_ids[:4096]

    input_tensor = torch.tensor(input_ids, dtype=torch.long, device=device)

    with torch.no_grad():
        outputs = model(input_tensor.unsqueeze(dim=0), output_hidden_states=True)
        f = outputs[1][-1]  # Last layer
        f_0 = outputs[1][0]  # Zero layer (embeddings)

        # Calculate loss positionally
        # Permute needed for CrossEntropyLoss: (Batch, Classes, Seq_Len) vs (Batch, Seq_Len)
        L = loss_fn(outputs[0].permute(0, 2, 1), input_tensor.unsqueeze(dim=0)).cpu().detach().numpy()

        current_len = L.shape[1]
        positional_losses[:current_len] += L[0, :]
        counter[:current_len] += 1.0

        # Store embeddings
        gf_embeddings_full[i, :] = torch.mean(f, dim=1).cpu().numpy()
        gf_embeddings_zero[i, :] = torch.mean(f_0, dim=1).cpu().numpy()

        # Store short context embeddings based on sequence length
        seq_len = f_0.shape[1]

        if seq_len > 500:
            gf_embeddings_short_500[i, :] = torch.mean(f_0[:, :500], dim=1).cpu().numpy()
            gf_embeddings_short_250[i, :] = torch.mean(f_0[:, :250], dim=1).cpu().numpy()
            gf_embeddings_short_125[i, :] = torch.mean(f_0[:, :125], dim=1).cpu().numpy()
            gf_embeddings_short_60[i, :] = torch.mean(f_0[:, :60], dim=1).cpu().numpy()
            gf_embeddings_short_30[i, :] = torch.mean(f_0[:, :30], dim=1).cpu().numpy()
        elif seq_len > 250:
            gf_embeddings_short_500[i, :] = torch.mean(f_0, dim=1).cpu().numpy()
            gf_embeddings_short_250[i, :] = torch.mean(f_0[:, :250], dim=1).cpu().numpy()
            gf_embeddings_short_125[i, :] = torch.mean(f_0[:, :125], dim=1).cpu().numpy()
            gf_embeddings_short_60[i, :] = torch.mean(f_0[:, :60], dim=1).cpu().numpy()
            gf_embeddings_short_30[i, :] = torch.mean(f_0[:, :30], dim=1).cpu().numpy()
        elif seq_len > 125:
            gf_embeddings_short_500[i, :] = torch.mean(f_0, dim=1).cpu().numpy()
            gf_embeddings_short_250[i, :] = torch.mean(f_0, dim=1).cpu().numpy()
            gf_embeddings_short_125[i, :] = torch.mean(f_0[:, :125], dim=1).cpu().numpy()
            gf_embeddings_short_60[i, :] = torch.mean(f_0[:, :60], dim=1).cpu().numpy()
            gf_embeddings_short_30[i, :] = torch.mean(f_0[:, :30], dim=1).cpu().numpy()
        elif seq_len > 60:
            gf_embeddings_short_500[i, :] = torch.mean(f_0, dim=1).cpu().numpy()
            gf_embeddings_short_250[i, :] = torch.mean(f_0, dim=1).cpu().numpy()
            gf_embeddings_short_125[i, :] = torch.mean(f_0, dim=1).cpu().numpy()
            gf_embeddings_short_60[i, :] = torch.mean(f_0[:, :60], dim=1).cpu().numpy()
            gf_embeddings_short_30[i, :] = torch.mean(f_0[:, :30], dim=1).cpu().numpy()
        elif seq_len > 30:
            gf_embeddings_short_500[i, :] = torch.mean(f_0, dim=1).cpu().numpy()
            gf_embeddings_short_250[i, :] = torch.mean(f_0, dim=1).cpu().numpy()
            gf_embeddings_short_125[i, :] = torch.mean(f_0, dim=1).cpu().numpy()
            gf_embeddings_short_60[i, :] = torch.mean(f_0, dim=1).cpu().numpy()
            gf_embeddings_short_30[i, :] = torch.mean(f_0[:, :30], dim=1).cpu().numpy()


plot_umap(gf_embeddings_short_500, adata.obs.cell_type, f'{Dataset}_gene_embedding_geneformer_zero_500', dpi=600)
plot_umap(gf_embeddings_short_250, adata.obs.cell_type, f'{Dataset}_gene_embedding_geneformer_zero_250', dpi=600)
plot_umap(gf_embeddings_short_125, adata.obs.cell_type, f'{Dataset}_gene_embedding_geneformer_zero_125', dpi=600)
plot_umap(gf_embeddings_short_60, adata.obs.cell_type, f'{Dataset}_gene_embedding_geneformer_zero_60', dpi=600)
plot_umap(gf_embeddings_short_30, adata.obs.cell_type, f'{Dataset}_gene_embedding_geneformer_zero_30', dpi=600)
plot_umap(gf_embeddings_zero, adata.obs.cell_type, f'{Dataset}_gene_embedding_geneformer_zero_full', dpi=600)
plot_umap(gf_embeddings_full, adata.obs.cell_type, f'{Dataset}_gene_embedding_geneformer', dpi=600)


print("Evaluating Geneformer Full...")
geneformer_full = evaluate_embeddings(gf_embeddings_full, adata.obs.cell_type)

print("Evaluating Geneformer Zero...")
geneformer_zero = evaluate_embeddings(gf_embeddings_zero, adata.obs.cell_type)
geneformer_zero_500 = evaluate_embeddings(gf_embeddings_short_500, adata.obs.cell_type)
geneformer_zero_250 = evaluate_embeddings(gf_embeddings_short_250, adata.obs.cell_type)
geneformer_zero_125 = evaluate_embeddings(gf_embeddings_short_125, adata.obs.cell_type)
geneformer_zero_60 = evaluate_embeddings(gf_embeddings_short_60, adata.obs.cell_type)
geneformer_zero_30 = evaluate_embeddings(gf_embeddings_short_30, adata.obs.cell_type)


# Save to Json 
res_dict = {}
res_dict['Dataset'] = Dataset  # Store dataset name in results
res_dict['ARI'] = dict()
res_dict['Sillhouette'] = dict()

res_dict['ARI']['GF'] = geneformer_full[0]
res_dict['Sillhouette']['GF'] = geneformer_full[1]

res_dict['ARI']['GF_0'] = geneformer_zero[0]
res_dict['Sillhouette']['GF_0'] = geneformer_zero[1]

res_dict['ARI']['GF_500'] = geneformer_zero_500[0]
res_dict['Sillhouette']['GF_500'] = geneformer_zero_500[1]

res_dict['ARI']['GF_250'] = geneformer_zero_250[0]
res_dict['Sillhouette']['GF_250'] = geneformer_zero_250[1]

res_dict['ARI']['GF_125'] = geneformer_zero_125[0]
res_dict['Sillhouette']['GF_125'] = geneformer_zero_125[1]

res_dict['ARI']['GF_60'] = geneformer_zero_60[0]
res_dict['Sillhouette']['GF_60'] = geneformer_zero_60[1]

res_dict['ARI']['GF_30'] = geneformer_zero_30[0]
res_dict['Sillhouette']['GF_30'] = geneformer_zero_30[1]

res_dict['ARI']['PCA'] = pca_perf[0]
res_dict['Sillhouette']['PCA'] = pca_perf[1]

res_dict['ARI']['Random_Emb'] = random_emb_gf[0]
res_dict['Sillhouette']['Random_Emb'] = random_emb_gf[1]

# Save to file
output_filename = f"{Dataset}_results.json"
with open(output_filename, 'w') as f:
    json.dump(res_dict, f, indent=4)

print(f"Results saved to {output_filename}")
