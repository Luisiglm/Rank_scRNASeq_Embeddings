import numpy as np



class untrained_ranked_embeddings:
    def __init__(self, d = 512, max_seq = 1000, no_genes = 20000):
        self.d = d
        self.max_seq = max_seq
        self.no_genes = no_genes
        # initialise gene embeddings
        self.gene_emb = np.random.normal(0,  1 / (d ** .5), (self.no_genes, d))
    def __call__(self, counts):
        return get_cell_embeddings(counts, self.d, self.max_seq, self.gene_emb)


def get_cell_embeddings(counts, d, max_seq, gene_emb):
    ranking = np.argsort(-counts, axis=1)
    cell_emb = np.zeros((counts.shape[0], d))
    for i in range(counts.shape[0]):
        mask = counts[i, ranking[i, :]] > 0.0
        if mask.any():
            if not mask.all():  # there is at least one zero
                first_zero = mask.argmin()
            else:
                first_zero = mask.shape[0]  # all expressed, no cutoff from zeros
            R_i = ranking[i, :min(first_zero, max_seq)]
            cell_emb[i, :] = np.mean(gene_emb[R_i, :], axis=0)
    return cell_emb