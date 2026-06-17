

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



start  = 0
end = batch

iterations = math.ceil(token_inputs.shape[0]/batch)

log_likelihood = np.zeros((4096,))
visits = np.zeros((4096,))
accuracy = np.zeros((4096,))
entropy = np.zeros((4096,))

def mask_tokens_get_y(token_inputs, token_dict, prob = 0.15):
    mask = (
        (torch.rand(token_inputs.shape) > (1 - prob)) &
        (token_inputs > 0) &
        (token_inputs != token_dict.get('<cls>')) &
        (token_inputs != token_dict.get('<eos>'))
    )
    x = token_inputs.clone()
    x[mask] = token_dict.get('<mask>')
    y = token_inputs.clone()
    y[~mask] = -100
    return x, y


loss = torch.nn.CrossEntropyLoss(reduction = 'none')
loss = loss.to(device)


from tqdm import tqdm
for i in tqdm(range(iterations)):
    input_ids = token_inputs[start:end, :]
    attention_mask = (input_ids != 0).long()
    max_len = attention_mask.sum(dim=1).max().item()
    input_ids = input_ids[:, :max_len]
    attention_mask = attention_mask[:, :max_len]
    x,y = mask_tokens_get_y(input_ids, token_dict)
    x = x.to(device)
    y = y.to(device)
    attention_mask = attention_mask.to(device)
    with torch.inference_mode():
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            outputs = model(
                input_ids=x,
                attention_mask=attention_mask,
                output_hidden_states=False
            )
            L = loss(outputs.logits.permute(0,2,1), y)
            visit_batch = torch.sum(y!=-100, dim = 0).detach().cpu().numpy()
            output_prob = torch.nn.functional.softmax(outputs.logits)
            entropy_batch = torch.log(output_prob)*output_prob # BxSxC
            entropy_batch = torch.sum(-entropy_batch, dim = -1)*(y!=-100) # BxS
            accuracy_batch = torch.sum(y == torch.argmax(outputs.logits, dim = -1), dim = 0) # BxS -> S
    L = np.sum(L.detach().cpu().numpy(), axis = 0)
    entropy[:L.shape[0]] += torch.sum(entropy_batch, dim = 0).detach().cpu().numpy() # BxS -> #S
    accuracy[:L.shape[0]] += accuracy_batch.detach().cpu().numpy()
    log_likelihood[:L.shape[0]] += L
    visits[:L.shape[0]] += visit_batch
    start += batch
    end += batch
    if end > token_inputs.shape[0]:
        end = token_inputs.shape[0]
    

np.savetxt(Dataset + 'entropy_geneformer_V2.txt', entropy)
np.savetxt(Dataset + 'accuracy_geneformer_V2.txt', accuracy)
np.savetxt(Dataset + 'log_lik_geneformer_V2.txt', log_likelihood)
np.savetxt(Dataset + 'visits_geneformer_V2.txt', visits)
