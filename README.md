
# Ranking is all you Need: Random Gene Embeddings Rival Large-Scale scRNA-seq Foundation Models
---

**Ellie Smartt¹, Feargus O'Gorman¹, Walter Kolch¹·², Vadim Zhernovkov¹ & Luis F. Iglesias-Martinez¹**

¹ Systems Biology Ireland, School of Medicine, University College Dublin, Dublin, Ireland  
² UCD Conway Institute of Biomolecular and Biomedical Research, University College Dublin, Dublin, Ireland

## Abstract

Foundation models have been introduced to the field of scRNA-seq. They are promising tools able to perform zero-shot classification, deal with batch effects and identify disease genes. Much of their success has been attributed to learning biological patterns after being trained on millions of cells. Several of these models use a preranked list of top expressed genes per cell as input. This introduces an important inductive bias, and we argue that it is enough to account for some of the milestones that scRNA-Seq foundation models have accomplished. Here, we show that using random gene embeddings on a pre-ranked gene list achieves similar results to models with hundreds of millions of parameters trained on millions of cells. We suggest that some of the tests used to benchmark scRNA-sSeq foundation models performance should be redesigned.  


## Repository Structure

```
Rank_scRNASeq_Embeddings/
├── README.md
├── LICENSE
├── CITATION.cff
├── sc_utils.toml          # installs sc_utils as a local package
├── .gitignore
├── environments/
│   ├── C2S_requirements.txt
│   ├── geneformer_requirements.txt
│   ├── scgpt_requirements.txt
│   ├── scFoundation_requirements.txt
│   └── untrained_embeddings_requirements.txt
├── src/
│   └── sc_utils/
│       ├── __init__.py
│       └── helpers.py      # evaluate_embeddings, plot_umap
├── scripts/
│   ├── c2s/
|   |   ├── C2S_possitional_loss.py
│   │   ├── C2S_script.py
│   │   └── C2S_ablation.py
│   ├── geneformer/
│   │   ├── geneformer_loglik.py
│   │   └── geneformer_cell_embeddings.py
│   ├── scgpt/
│   │   └── scGPT_script.py
│   ├── scfoundation/
│   │   └── scFoundation_script.py
│   └── untrained_embeddings/
│       └── evaluate.py
├── data/
│   └── README.md           # instructions to download datasets

```

## Installation

Each model requires its own virtual environment due to incompatible dependencies.
`sc_utils` must be installed in each environment to share evaluation and plotting utilities.

### C2S

```bash
python -m venv envs/c2s
source envs/c2s/bin/activate
pip install -r environments/c2s_requirements.txt
pip install -e .
```

### Geneformer

```bash
python -m venv envs/geneformer
source envs/geneformer/bin/activate
pip install -r environments/geneformer_requirements.txt
pip install -e .
```

### scGPT

```bash
python -m venv envs/scgpt
source envs/scgpt/bin/activate
pip install -r environments/scgpt_requirements.txt
pip install -e .
```

### scFoundation

```bash
python -m venv envs/scfoundation
source envs/scfoundation/bin/activate
pip install -r environments/scfoundation_requirements.txt
pip install -e .
```

### Untrained Embeddings

```bash
python -m venv envs/untrained_embeddings
source envs/untrained_embeddings/bin/activate
pip install -r environments/untrained_embeddings_requirements.txt
pip install -e .
```

> **Note for Windows users:** replace `source envs/<name>/bin/activate`
> with `envs\<name>\Scripts\activate`.
