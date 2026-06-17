import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
import scanpy as sc
from sklearn.metrics import adjusted_rand_score, silhouette_score, normalized_mutual_info_score
from sklearn.cluster import KMeans
from tqdm import tqdm
from geneformer import TranscriptomeTokenizer
from transformers import AutoModelForMaskedLM
import torch
import math


parser = argparse.ArgumentParser(description="Random Embeddings Analysis")

parser.add_argument('--address', type=str, required=True, help='Path to the h5ad file')
parser.add_argument('--batch', type=int, required=False,   help='batch size')
parser.add_argument('--max_seq', type=int, required=False,   help='maximum gene sequence length')

args = parser.parse_args()


address = args.address
max_seq = args.max_seq
batch = args.batch


print(f"Loading file from: {address}")


# Read the h5ad file using the provided address
adata = sc.read_h5ad(address)

raw_title = adata.uns.get('title', 'Unknown_Dataset') # Default if title missing
# Replace spaces and slashes to make it safe for filenames
Dataset = str(raw_title).replace(" ", "_").replace("/", "-")


x = adata.X.todense()
ranking = np.argsort(-x, axis=1)


tokenizer = TranscriptomeTokenizer()


# If you intended to save the processed adata, uncomment the line below:
# adata.write(address)

model = AutoModelForMaskedLM.from_pretrained("ctheodoris/Geneformer")
model.eval()

d_GF = model.bert.embeddings.word_embeddings.weight.shape[1]

gene_ids = adata.var_names.tolist()

# Initialize arrays for embeddings
gf_embeddings_zero = np.zeros((x.shape[0], d_GF))
gf_embeddings_full = np.zeros((x.shape[0], d_GF))
gf_embeddings_short_1000 = np.zeros((x.shape[0], d_GF))
gf_embeddings_short_125 = np.zeros((x.shape[0], d_GF))
gf_embeddings_short_250 = np.zeros((x.shape[0], d_GF))
gf_embeddings_short_500 = np.zeros((x.shape[0], d_GF))


positional_losses = np.zeros((max_seq,))
counter = np.zeros((max_seq,))

loss_fn = torch.nn.CrossEntropyLoss(reduce=None, ignore_index = 0,reduction='none')
device = 'cuda:1' if torch.cuda.is_available() else 'cpu'

model = model.to(device)
loss_fn = loss_fn.to(device)

iterations = math.ceil(x.shape[0]/batch)

start = 0
end = batch

ranking = np.asarray(ranking)
x = np.asarray(x)


# Run the Geneformer
for i in tqdm(range(iterations)):
    # sample from x
    rows = np.arange(start, end)[:, None]     # shape (B,1)
    mask = x[rows, ranking[start:end, :]] == 0.0
    ranking_batch = ranking[start:end,:]
    cols = mask.argmax(axis=1)
    input_tensor = torch.zeros((mask.shape[0],max_seq), dtype = torch.long)+tokenizer.gene_token_dict.get('<pad>')
    max_k = 0
    for j in range(mask.shape[0]):
        R_i = ranking_batch[j,:cols[j]]
        sorted_genes = [gene_ids[idx] for idx in R_i]
        input_ids = [tokenizer.gene_token_dict.get('<cls>')] + \
                [tokenizer.gene_token_dict.get(gene) for gene in sorted_genes]
        input_ids = [val for val in input_ids if val is not None]
        if len(input_ids) > (max_seq-1):
            input_ids = input_ids[:max_seq-1] 
        input_ids += [tokenizer.gene_token_dict.get('<eos>')]
        input_tensor[j,:len(input_ids)] = torch.tensor(input_ids, dtype = torch.long)
        if len(input_ids)>max_k:
            max_k = len(input_ids)
    input_tensor = input_tensor[:,:max_k]
    pad_mask = (input_tensor != tokenizer.gene_token_dict.get('<pad>'))
    input_tensor = input_tensor.to(device=device)
    attention_mask = pad_mask.long().to(device=device)
    with torch.no_grad():
        outputs = model(input_tensor, attention_mask = attention_mask, output_hidden_states=True)
        f = outputs[1][-2]  # Second Last layer
        f_0 = outputs[1][0]  # Zero layer (embeddings)
        # Calculate loss positionally
        # Permute needed for CrossEntropyLoss: (Batch, Classes, Seq_Len) vs (Batch, Seq_Len)
        L = loss_fn(outputs[0].permute(0, 2, 1), input_tensor).cpu().detach().numpy()
        current_len = L.shape[1]
        positional_losses[:current_len] += np.sum(L, axis = 0)
        counter[:current_len] += L.shape[0]
        # Store embeddings
        gf_embeddings_full[start:end, :] = torch.mean(f, dim=1).cpu().numpy()
        gf_embeddings_zero[start:end, :] = torch.mean(f_0, dim=1).cpu().numpy()
        # Store short context embeddings based on sequence length
        seq_len = f_0.shape[1]
        if seq_len > 1000:
            gf_embeddings_short_1000[start:end, :] = torch.mean(f_0[:, :1000], dim=1).cpu().numpy()
            gf_embeddings_short_500[start:end, :] = torch.mean(f_0[:, :500], dim=1).cpu().numpy()
            gf_embeddings_short_250[start:end, :] = torch.mean(f_0[:, :250], dim=1).cpu().numpy()
            gf_embeddings_short_125[start:end, :] = torch.mean(f_0[:, :125], dim=1).cpu().numpy()
        elif seq_len > 500:
            gf_embeddings_short_1000[start:end, :] = torch.mean(f_0, dim=1).cpu().numpy()
            gf_embeddings_short_500[start:end, :] = torch.mean(f_0[:, :500], dim=1).cpu().numpy()
            gf_embeddings_short_250[start:end, :] = torch.mean(f_0[:, :250], dim=1).cpu().numpy()
            gf_embeddings_short_125[start:end, :] = torch.mean(f_0[:, :125], dim=1).cpu().numpy()
        elif seq_len > 250:
            gf_embeddings_short_1000[start:end, :] = torch.mean(f_0, dim=1).cpu().numpy()
            gf_embeddings_short_500[start:end, :] = torch.mean(f_0, dim=1).cpu().numpy()
            gf_embeddings_short_250[start:end, :] = torch.mean(f_0[:, :250], dim=1).cpu().numpy()
            gf_embeddings_short_125[start:end, :] = torch.mean(f_0[:, :125], dim=1).cpu().numpy()
        elif seq_len > 125:
            gf_embeddings_short_1000[start:end, :] = torch.mean(f_0, dim=1).cpu().numpy()
            gf_embeddings_short_500[start:end, :] = torch.mean(f_0, dim=1).cpu().numpy()
            gf_embeddings_short_250[start:end, :] = torch.mean(f_0, dim=1).cpu().numpy()
            gf_embeddings_short_125[start:end, :] = torch.mean(f_0[:, :125], dim=1).cpu().numpy()
        else:
            gf_embeddings_short_1000[start:end, :] = torch.mean(f_0, dim=1).cpu().numpy()
            gf_embeddings_short_500[start:end, :] = torch.mean(f_0, dim=1).cpu().numpy()
            gf_embeddings_short_250[start:end, :] = torch.mean(f_0, dim=1).cpu().numpy()
            gf_embeddings_short_125[start:end, :] = torch.mean(f_0, dim=1).cpu().numpy()
    start = start + batch
    end = end + batch
    if end>x.shape[0]:
        end = x.shape[0]
       
positional_losses = positional_losses/counter

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


ari_leiden_4096, nmi_leiden_4096 = plot_umap(gf_embeddings_full, adata.obs["cell_type"], title = Dataset, filename = Dataset+'_Geneformer_' + str(d_GF))
ari_leiden_0, nmi_leiden_0 = plot_umap(gf_embeddings_zero, adata.obs["cell_type"],title = Dataset, filename =Dataset +'_Geneformer_Zero_' +str( d_GF))
ari_leiden_1000, nmi_leiden_1000 = plot_umap(gf_embeddings_short_1000, adata.obs["cell_type"], title = Dataset,filename =Dataset+ 'Geneformer_Zero_Top_1000_' + str(d_GF))
ari_leiden_500, nmi_leiden_500 = plot_umap(gf_embeddings_short_500, adata.obs["cell_type"],title = Dataset, filename = Dataset+'Geneformer_Zero_Top_500_' + str(d_GF))
ari_leiden_250, nmi_leiden_250 = plot_umap(gf_embeddings_short_250, adata.obs["cell_type"],title = Dataset, filename =Dataset +'Geneformer_Zero_Top_250_' + str(d_GF))
ari_leiden_125, nmi_leiden_125 = plot_umap(gf_embeddings_short_125, adata.obs["cell_type"], title = Dataset,filename =Dataset +'Geneformer_Zero_Top_125_' + str(d_GF))

# write results
results = {}
results['4096'] = {'ARI':ari_leiden_4096, 'NMI':nmi_leiden_4096}
results['0'] = {'ARI':ari_leiden_0, 'NMI':nmi_leiden_0}
results['1000'] = {'ARI':ari_leiden_1000, 'NMI':nmi_leiden_1000}
results['500'] = {'ARI':ari_leiden_500, 'NMI':nmi_leiden_500}
results['250'] = {'ARI':ari_leiden_250, 'NMI':nmi_leiden_250}
results['125'] = {'ARI':ari_leiden_125, 'NMI':nmi_leiden_125}
results['Positional_Loss'] = list(positional_losses)
# Save to file
output_filename = f"{Dataset}_Geneformer_{d_GF}_results.json"
with open(output_filename, 'w') as f:
    json.dump(results, f, indent=4)

print(f"Results saved to {output_filename}")




