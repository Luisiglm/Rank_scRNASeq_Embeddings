# Taken from: https://github.com/vandijklab/cell2sentence/blob/master/tutorials/c2s_tutorial_2_cell_embedding.ipynb

# Python built-in libraries
import os
import random
from collections import Counter
import argparse

# Third-party libraries
import numpy as np
from tqdm import tqdm

# Single-cell libraries
import anndata
import scanpy as sc

# Cell2Sentence imports
import cell2sentence as cs
from cell2sentence.tasks import embed_cells

from sc_utils import evaluate_embeddings, plot_umap


# Read Dataset

parser = argparse.ArgumentParser(description="Run Geneformer Analysis")

parser.add_argument('--address', type=str, required=True, help='Path to the h5ad file')

args = parser.parse_args()


address = args.address

print(f"Loading file from: {address}")

# Read the h5ad file using the provided address
adata = sc.read_h5ad(address)




adata_obs_cols_to_keep = ["cell_type"]

# Change gene names to Symbol


adata.var.index = adata.var['feature_name']
# Ensure they are unique
adata.var.index = adata.var.index.astype(str)
adata.var_names_make_unique()

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

# Define CSModel object

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load model directly from Hugging Face Hub
model_id = "vandijklab/C2S-Scale-Gemma-2-2B"

model = AutoModelForCausalLM.from_pretrained(
    model_id,
).to(device)



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

cell_embeddings = []

for i in tqdm(range(N)):
    samples = arrow_ds[start:end]
    cell_sentences = samples["cell_sentence"]
    cell_embeddings.append(embed_cells_batched(model, tokenizer, cell_sentences))
    start += batch_size
    end += batch_size
    if end > adata.X.shape[0]:
        end = adata.X.shape[0]


cell_emb = np.zeros((adata.X.shape[0], 2304))

j = 0

for i in cell_embeddings:
    for k in i:
        cell_emb[j,:] = k
        j +=1

adata.obsm["c2s_cell_embeddings"] = cell_emb

plot_umap(cell_emb, adata.obs.cell_type, f'{Dataset}_C2S', dpi=600)

# Create Dictionary to Store Results
res_dict = {}
res_dict['Dataset'] = Dataset
res_dict['ARI'] = dict()
res_dict['Sillhouette'] = dict()

C2S_perf = evaluate_embeddings(cell_emb, adata.obs.cell_type)

res_dict['ARI']['C2S'] = C2S_perf[0]
res_dict['Sillhouette']['C2S'] = C2S_perf[1]

# Save to file
output_filename = f"{Dataset}_C2S_results.json"
with open(output_filename, 'w') as f:
    json.dump(res_dict, f, indent=4)

print(f"Results saved to {output_filename}")
