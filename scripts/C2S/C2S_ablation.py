# Taken from: https://github.com/vandijklab/cell2sentence/blob/master/tutorials/c2s_tutorial_2_cell_embedding.ipynb

# Python built-in libraries
import re
import random
import argparse

import numpy as np
from tqdm import tqdm

import anndata
import scanpy as sc

# Cell2Sentence imports
import cell2sentence as cs
from sc_utils import plot_umap

from cell2sentence.prompt_formatter import C2SPromptFormatter
# AI packages
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import matplotlib.pyplot as plt
# Import our utils modules

from sc_utils import  plot_umap
from sklearn.metrics import adjusted_rand_score, silhouette_score,normalized_mutual_info_score

# Read Dataset

parser = argparse.ArgumentParser(description="Run C2S Ablation Test")

parser.add_argument('--address', type=str, required=True, help='Path to the h5ad file')
parser.add_argument('--context', type=str, required=True, help='top k genes')

args = parser.parse_args()
context = int(args.context)


address = args.address

print(f"Loading file from: {address}")

# Read the h5ad file using the provided address
adata = sc.read_h5ad(address)
adata.obs['organism'] ='Homo sapiens'

# Change gene names to Symbol
adata.var.index = adata.var['feature_name']
# Ensure they are unique
adata.var.index = adata.var.index.astype(str)
adata.var_names_make_unique()
# remove genes that have no gene symbol
adata_obs_cols_to_keep = ["cell_type", "organism"]
genes_to_keep = []


for  i in range(adata.var.shape[0]):
    match = re.search(r"^([^.]+)", adata.var.ensembl_id[i])
    if match.group(1) != adata.var.feature_name[i]:
        genes_to_keep.append(i)


adata = adata[:,genes_to_keep]

# Set device
device = "cpu"

# Load model directly from Hugging Face Hub
model_id = "vandijklab/C2S-Scale-Gemma-2-2B"

tokenizer = AutoTokenizer.from_pretrained(model_id)

# Change to right sided tokenizer

tokenizer.padding_side = "right"

model = AutoModelForCausalLM.from_pretrained(
    model_id,
).to(device)


gene_weights = (
    model.model.embed_tokens.weight
    .detach()
    .to(torch.float32)   # or .float()
    .cpu()
    .numpy()
)

N = adata.X.shape[0]

z_emb = np.zeros((N,2034 ))

SEED = 1234
random.seed(SEED)
np.random.seed(SEED)

# Create CSData object
arrow_ds, vocabulary = cs.CSData.adata_to_arrow(
    adata=adata,
    random_state=SEED,
    sentence_delimiter=' ',
    label_col_names=adata_obs_cols_to_keep
)

# Read Study Title
raw_title = adata.uns.get('title', 'Unknown_Dataset') # Default if title missing
# Replace spaces and slashes to make it safe for filenames
Dataset = str(raw_title).replace(" ", "_").replace("/", "-")

n_genes = 1000
pad_token_id = tokenizer.pad_token_id

prompt_formatter = C2SPromptFormatter(task="cell_type_generation", top_k_genes=n_genes)
# format from arrow_ds
formatted_hf_ds = prompt_formatter.format_hf_ds(arrow_ds)




def embed_responses(formatted_hf_ds, gene_weights, tokenizer, context, chunk_size=2000):
    N = formatted_hf_ds.num_rows
    E = gene_weights.shape[1]  # 2034
    z_emb = np.empty((N, E), dtype=np.float32)
    for start in tqdm(range(0, N, chunk_size)):
        end = min(start + chunk_size, N)
        responses = [formatted_hf_ds[i]["response"] for i in range(start, end)]
        encoded = tokenizer(
            responses,
            add_special_tokens=False,
            padding=True,
            truncation=True,        # truncate to max_length
            max_length=context,     # context = max tokens per sequence
            return_tensors="np"
        )
        input_ids = encoded["input_ids"]       # (chunk, seq_len)
        attn_mask = encoded["attention_mask"]  # (chunk, seq_len)
        embeddings = gene_weights[input_ids]   # (chunk, seq_len, E)
        embeddings *= attn_mask[:, :, None]
        counts = np.maximum(attn_mask.sum(axis=1, keepdims=True), 1)
        z_emb[start:end] = embeddings.sum(axis=1) / counts
    return z_emb


z_emb = embed_responses(formatted_hf_ds, gene_weights, tokenizer, context, chunk_size=256)

ari_leiden, nmi_leiden = plot_umap(z_emb, adata.obs['cell_type'], Dataset + str(context) + '_C2S_ablation')

np.savetxt(Dataset + str(context) + '_C2S_ablation.csv', np.array((ari_leiden, nmi_leiden)))
