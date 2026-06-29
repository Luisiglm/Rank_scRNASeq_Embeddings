

import json
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForMaskedLM
from geneformer import TranscriptomeTokenizer
from transformers import DataCollatorWithPadding
import numpy as np
import math
import matplotlib.pyplot as plt
import scanpy as sc
from sklearn.metrics import  adjusted_rand_score, normalized_mutual_info_score
import argparse

from geneformer import GENE_MEDIAN_FILE, TOKEN_DICTIONARY_FILE
import pickle
from sc_utils import plot_umap

# Geneformer Ranking and tokenization

with open(TOKEN_DICTIONARY_FILE, "rb") as f:
    token_dict = pickle.load(f)


with open(GENE_MEDIAN_FILE, "rb") as f:
    median_dict = pickle.load(f)


parser = argparse.ArgumentParser(description="Random Embeddings Analysis")

parser.add_argument('--address', type=str, required=True, help='Path to the h5ad file')
parser.add_argument('--batch', type=int, required=False,   help='batch size')


args = parser.parse_args()

address = args.address
batch = args.batch


print(f"Loading file from: {address}")


# Read the h5ad file using the provided address
adata = sc.read_h5ad(address)

raw_title = adata.uns.get('title', 'Unknown_Dataset') # Default if title missing
# Replace spaces and slashes to make it safe for filenames
Dataset = str(raw_title).replace(" ", "_").replace("/", "-")


# Get raw counts

raw_counts = np.asarray(adata.layers['decontXcounts'].todense())

ensembl_ids = adata.var.axes[0]

# keep ensembl_ids in token_dict

keepers = []

for i in range(len(ensembl_ids)):
    if token_dict.get(ensembl_ids[i]) is not None:
        keepers.append(i)

counts_filter = raw_counts[:,keepers]
ensembl_ids_filter = ensembl_ids[keepers]

ensembl_token_index = [token_dict[i] for i in ensembl_ids_filter]
ensembl_token_tensor = torch.tensor(ensembl_token_index, dtype=torch.long)
n_counts = np.sum(counts_filter, axis=1, keepdims=True)
# normalise to 10000 reads
counts_filter = counts_filter / n_counts*10000

for i in range(len(ensembl_ids_filter)):
    counts_filter[:,i] = counts_filter[:,i]/median_dict[ensembl_ids_filter[i]]

gene_ranking = np.argsort(-counts_filter, axis = 1)


token_inputs = torch.zeros((counts_filter.shape[0],4096), dtype=torch.long)

# Reorder zero_mask to match ranking order
for i in range(gene_ranking.shape[0]):
    x = counts_filter[i,gene_ranking[i,:]]
    zero_mask = (x == 0)
    first_zero = np.argmax(zero_mask) if zero_mask.any() else len(x)
    if first_zero >  4094:
        first_zero = 4094
    token_inputs[i, 0] = token_dict['<cls>']
    token_inputs[i, 1 : first_zero + 1] = ensembl_token_tensor[gene_ranking[i, :first_zero]]
    token_inputs[i, first_zero + 1] = token_dict['<eos>']


# Model Load

model = AutoModelForMaskedLM.from_pretrained("ctheodoris/Geneformer")
model.eval()

device = 'cuda:0'

model = model.to(device)

loss = torch.nn.CrossEntropyLoss()

batch_size = 50

start  = 0
end = batch_size

iterations = math.ceil(token_inputs.shape[0]/batch_size)

embeddings         = np.zeros((token_inputs.shape[0], 1152), dtype=np.float16)
embeddings_0       = np.zeros((token_inputs.shape[0], 1152), dtype=np.float16)
embeddings_0_1000  = np.zeros((token_inputs.shape[0], 1152), dtype=np.float16)
embeddings_0_500   = np.zeros((token_inputs.shape[0], 1152), dtype=np.float16)
embeddings_0_125   = np.zeros((token_inputs.shape[0], 1152), dtype=np.float16)

from tqdm import tqdm
for i in tqdm(range(iterations)):
    input_ids = token_inputs[start:end, :]
    attention_mask = (input_ids != 0).long()
    max_len = attention_mask.sum(dim=1).max().item()
    input_ids = input_ids[:, :max_len]
    attention_mask = attention_mask[:, :max_len]
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    with torch.inference_mode():
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,  # Added missing comma
                output_hidden_states=True  # Corrected parameter name
            )
        # For classification: logits
        hidden_state = outputs.hidden_states[-2]
        layer_0 = outputs.hidden_states[0]
        del outputs  # frees all other hidden states
        # mask out <cls>, <eos>, and <pad>
        emb_mask = attention_mask * (input_ids != token_dict.get('<cls>')) * (input_ids != token_dict.get('<eos>'))
        # Add hidden dimension for broadcasting: shape becomes (B, L, 1)
        emb_mask = emb_mask.unsqueeze(-1)
        # Calculate sum of valid token embeddings and divide by valid token counts
        sum_embeddings = torch.sum(hidden_state * emb_mask, dim=1)
        valid_counts = emb_mask.sum(dim=1).clamp(min=1e-9)  # clamp to prevent division by zero
        sum_embeddings_0 = torch.sum(layer_0 * emb_mask, dim=1)
        sum_embeddings_0_1000 = torch.sum(layer_0[:,:1000] * emb_mask[:,:1000], dim=1)
        valid_counts_1000 = emb_mask[:,:1000].sum(dim=1).clamp(min=1e-9)  # clamp to prevent division by zero
        sum_embeddings_0_500 = torch.sum(layer_0[:, :500] * emb_mask[:, :500], dim=1)
        valid_counts_500 = emb_mask[:, :500].sum(dim=1).clamp(min=1e-9)  # clamp to prevent division by zero
        sum_embeddings_0_125 = torch.sum(layer_0[:, :125] * emb_mask[:, :125], dim=1)
        valid_counts_125 = emb_mask[:, :125].sum(dim=1).clamp(min=1e-9)  # clamp to prevent division by zero
    embeddings[start:end, :] = (sum_embeddings / valid_counts).cpu().numpy()
    embeddings_0[start:end, :] = (sum_embeddings_0 / valid_counts).cpu().numpy()
    embeddings_0_1000[start:end, :] = (sum_embeddings_0_1000 / valid_counts_1000 ).cpu().numpy()
    embeddings_0_500[start:end, :] = (sum_embeddings_0_500 / valid_counts_500 ).cpu().numpy()
    embeddings_0_125[start:end, :] = (sum_embeddings_0_125 / valid_counts_125 ).cpu().numpy()
    del hidden_state, layer_0, emb_mask
    del sum_embeddings, sum_embeddings_0
    del sum_embeddings_0_1000, sum_embeddings_0_500, sum_embeddings_0_125
    del valid_counts, valid_counts_1000, valid_counts_500, valid_counts_125
    del input_ids, attention_mask
    torch.cuda.empty_cache()
    start += batch_size
    end += batch_size
    if end > token_inputs.shape[0]:
        end = token_inputs.shape[0]



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


ari, nmi = plot_umap(embeddings, adata.obs.cell_type, f'{Dataset}_Geneformer_V2_', dpi=600)
ari_0, nmi_0 = plot_umap(embeddings_0, adata.obs.cell_type, f'{Dataset}_Geneformer_V2_0', dpi=600)
ari_0_125, nmi_0_125 = plot_umap(embeddings_0_125, adata.obs.cell_type, f'{Dataset}_Geneformer_V2_0_125', dpi=600)
ari_0_500, nmi_0_500 = plot_umap(embeddings_0_500, adata.obs.cell_type, f'{Dataset}_Geneformer_V2_0_500', dpi=600)
ari_0_1000, nmi_0_1000 = plot_umap(embeddings_0_1000, adata.obs.cell_type, f'{Dataset}_Geneformer_V2_0_1000', dpi=600)


# Create Dictionary to Store Results
res_dict = {}
res_dict['ARI'] = ari
res_dict['NMI'] = nmi
res_dict['ARI_0'] = ari_0
res_dict['NMI_0'] = nmi_0
res_dict['ARI_0_125'] = ari_0_125
res_dict['NMI_0_125'] = nmi_0_125
res_dict['ARI_0_500'] = ari_0_500
res_dict['NMI_0_500'] = nmi_0_500
res_dict['ARI_0_1000'] = ari_0_1000
res_dict['NMI_0_1000'] = nmi_0_1000
# Save to file
output_filename = f"{Dataset}_Geneformer_V2_results.json"
with open(output_filename, 'w') as f:
    json.dump(res_dict, f, indent=4)

print(f"Results saved to {output_filename}")