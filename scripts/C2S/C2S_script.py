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
from sklearn.metrics import adjusted_rand_score, silhouette_score, normalized_mutual_info_score

# Import our utils modules

from sc_utils import  plot_umap
import matplotlib.pyplot as plt
 
import json

# Read Dataset

parser = argparse.ArgumentParser(description="Run C2S Analysis")

parser.add_argument('--address', type=str, required=True, help='Path to the h5ad file')

args = parser.parse_args()


address = args.address

print(f"Loading file from: {address}")

# Read the h5ad file using the provided address
adata = sc.read_h5ad(address)
adata.obs['organism'] ='Homo sapiens'

n_genes = 200

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
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load model directly from Hugging Face Hub
model_id = "vandijklab/C2S-Scale-Gemma-2-2B"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
).to(device)

# Keep genes that match the vocab in the tokenizer.


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


prompt_formatter = C2SPromptFormatter(task="cell_type_prediction", top_k_genes=n_genes)
# format from arrow_ds
formatted_hf_ds = prompt_formatter.format_hf_ds(arrow_ds)


def embed_cells_batched( model, tokenizer, prompt_list, max_num_tokens = 1024):
    """
    Embed multiple cell in batched fashion using the model, starting with a given prompt.
    Arguments:
        model: a C2S model for cell embedding
        prompt_list: a list of textual prompts
        max_num_tokens: the maximum number of tokens to generate given the model supplied
    Return:
        Text corresponding to the number `n` of tokens requested
    """
    tokens = tokenizer(prompt_list, padding=True, return_tensors='pt')
    input_ids = tokens['input_ids'].to(model.device)
    attention_mask = tokens['attention_mask'].to(model.device)
    if input_ids.shape[1] > max_num_tokens:
        input_ids = input_ids[:, :max_num_tokens]
        attention_mask = attention_mask[:, :max_num_tokens]
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
    # Take last layer output, average over sequence dimension
    all_embeddings = []
    for idx in range(len(prompt_list)):
        embedding = outputs.hidden_states[-1][idx].mean(0).float().detach().cpu().numpy()
        all_embeddings.append(embedding)
    return all_embeddings


batch_size = 4
start = 0
end = batch_size
N =  int(np.ceil(adata.X.shape[0]/batch_size))

embedded_cells = []
inference_batch_size = 4
batch_inputs = []
idx = 0
for sample_idx in tqdm(range(formatted_hf_ds.num_rows)):
    # Prepare inputs
    sample = formatted_hf_ds[sample_idx]
    model_input_prompt_str = sample["model_input"]
    batch_inputs.append(model_input_prompt_str)
    idx += 1
    # Inference on a batch of inputs
    if (len(batch_inputs) == inference_batch_size) or (idx == formatted_hf_ds.num_rows):
        cell_embeddings = embed_cells_batched(model, tokenizer, prompt_list=batch_inputs)
        embedded_cells += cell_embeddings
        batch_inputs = []

embedded_cells = np.stack(embedded_cells)





ari, nmi = plot_umap(embedded_cells, adata.obs.cell_type, f'{Dataset}_C2S', dpi=600)

# Create Dictionary to Store Results
res_dict = {}
res_dict['ARI'] = ari
res_dict['NMI'] = nmi


# Save to file
output_filename = f"{Dataset}_C2S_results.json"
with open(output_filename, 'w') as f:
    json.dump(res_dict, f, indent=4)

print(f"Results saved to {output_filename}")
