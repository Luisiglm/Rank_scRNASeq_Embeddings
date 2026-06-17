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

from cell2sentence.prompt_formatter import C2SPromptFormatter
# AI packages
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch



# Read Dataset

parser = argparse.ArgumentParser(description="Run C2S Positional Loss")

parser.add_argument('--address', type=str, required=True, help='Path to the h5ad file')

args = parser.parse_args()


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
device = "cuda:1" if torch.cuda.is_available() else "cpu"

# Load model directly from Hugging Face Hub
model_id = "vandijklab/C2S-Scale-Gemma-2-2B"

tokenizer = AutoTokenizer.from_pretrained(model_id)

# Change to right sided tokenizer

tokenizer.padding_side = "right"

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

n_genes = 1000
pad_token_id = tokenizer.pad_token_id

prompt_formatter = C2SPromptFormatter(task="cell_type_generation", top_k_genes=n_genes)
# format from arrow_ds
formatted_hf_ds = prompt_formatter.format_hf_ds(arrow_ds)

loss = torch.nn.CrossEntropyLoss(reduction = 'none')


loss = loss.to(device)

batch_size = 4
start = 0
end = batch_size
N =  int(np.ceil(adata.X.shape[0]/batch_size))

embedded_cells = []
inference_batch_size = 4
batch_inputs = []
batch_labels = []
idx = 0

positional_losses = np.zeros((1024))
visits =np.zeros((1024,))

for sample_idx in tqdm(range(formatted_hf_ds.num_rows)):
    # Prepare inputs
    sample = formatted_hf_ds[sample_idx]
    model_input_prompt_str = sample["model_input"]
    prompt_ids = tokenizer(model_input_prompt_str, add_special_tokens=True)["input_ids"]
    model_response = sample["response"]
    response_ids = tokenizer(model_response, add_special_tokens=False)["input_ids"]
    batch_inputs.append(prompt_ids + response_ids)
    idx += 1
    batch_labels.append([-100] * len(prompt_ids) + response_ids)
    if (len(batch_inputs) == inference_batch_size) or (idx == formatted_hf_ds.num_rows):
        # 1. Initialize tensors with pad_token_id for inputs, and -100 for labels
        token_tensor = torch.full((len(batch_inputs), 1024), pad_token_id, dtype=torch.long)
        labels_tensor = torch.full((len(batch_inputs), 1024), -100, dtype=torch.long)
        # 2. Fill the tensors correctly, matching the sequence lengths
        for j in range(len(batch_inputs)):
            seq_len = min(1024, len(batch_inputs[j]))
            token_tensor[j, :seq_len] = torch.tensor(batch_inputs[j][:seq_len])
            labels_tensor[j, :seq_len] = torch.tensor(batch_labels[j][:seq_len])
        # 3. Correct attention mask (1 for real tokens, 0 for padding)
        attention_mask = (token_tensor != pad_token_id).long()
        token_tensor = token_tensor.to(device)
        labels_tensor = labels_tensor.to(device)
        attention_mask = attention_mask.to(device)
        with torch.no_grad():
            outputs = model(
                input_ids=token_tensor,
                attention_mask=attention_mask,
                output_hidden_states=False
            )
            logits = outputs.logits  # (batch, seq_len, vocab_size)
            # 4. MANUALLY SHIFT logits and labels for next-token prediction
            # Logits: drop the last prediction. Labels: drop the first token.
            shift_logits = logits[..., :-1, :].contiguous()  # Shape: (batch, 1023, vocab)
            shift_labels = labels_tensor[..., 1:].contiguous()  # Shape: (batch, 1023)
            # Calculate positional loss (CrossEntropy expects (batch, vocab, seq_len))
            L = loss(shift_logits.permute(0, 2, 1), shift_labels).to('cpu')  # Shape: (batch, 1023)
        # 5. Accumulate losses and visits
        # We pad L and the label mask back to 1024 so it perfectly aligns with your numpy arrays
        # The last position is 0 because we can't predict what comes after the 1024th token
        padded_L = torch.zeros((len(batch_inputs), 1024))
        padded_L[:, :1023] = L
        padded_shift_labels = torch.full((len(batch_inputs), 1024), -100)
        padded_shift_labels[:, :1023] = shift_labels.cpu()
        positional_losses += torch.sum(padded_L, axis=0).numpy()
        # Count visits only where the shifted label is NOT -100
        visits += torch.sum(padded_shift_labels != -100, axis=0).numpy()
        # Reset batches
        batch_inputs = []
        batch_labels = []





np.savetxt(Dataset+'_C2S_positional_loss.txt', positional_losses )
np.savetxt(Dataset+'_C2S_positional_visits.txt', visits )


