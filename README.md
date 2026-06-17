
# Ranking is all you Need: Random Gene Embeddings Rival Large-Scale scRNA-seq Foundation Models
---

**Ellie Smartt¹, Feargus O'Gorman¹, Walter Kolch¹·², Vadim Zhernovkov¹ & Luis F. Iglesias-Martinez¹**

¹ Systems Biology Ireland, School of Medicine, University College Dublin, Dublin, Ireland  
² UCD Conway Institute of Biomolecular and Biomedical Research, University College Dublin, Dublin, Ireland

## Abstract

Foundation models have been introduced to the field of scRNA-seq. They are promising tools able to perform zero-shot classification, deal with batch effects and identify disease genes. Much of their success has been attributed to learning biological patterns after being trained on millions of cells. Several of these models use a preranked list of top expressed genes per cell as input. This introduces an important inductive bias, and we argue that it is enough to account for some of the milestones that scRNA-Seq foundation models have accomplished. Here, we show that using random gene embeddings on a pre-ranked gene list achieves similar results to models with hundreds of millions of parameters trained on millions of cells. We suggest that some of the tests used to benchmark scRNA-sSeq foundation models performance should be redesigned.  



