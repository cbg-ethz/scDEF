In "The Spatial and Genomic Hierarchy of Tumor Ecosystems Revealed by Single-Cell Technologies", Eric A. Smith and H. Courtney Hodges describe a hierarchy of tumor heterogeneity in which populations are defined by their cell type at one level, and by phenotypic states at another, more specific, level. For example, cancer cells can be grouped based on their progression in the epithelial-mesenchymal transition, or their ability to promote angiogenesis. Usually, such cancer-specific cell states are investigated via pathway enrichment analyses, where the top expressed genes within cells (or groups thereof) are compared with a database of known gene sets to find overrepresented sets. Another example is the distinct set of states associated with immune cell activation, as seen in tumor associated macrophages (TAMs), which have different expression patterns and functions based on the pathway of their activation.

Factorization methods can be used to identify gene sets. However, these methods assume independence between the factors, which may not hold in practice. For example, it has been observed that activation of the MAPK pathway is mutually exclusive with the epithelial-mesenchymal transition. Accounting for such possible relationships may facilitate the inference of accurate gene signatures.

Additionally, some groupings of cells may not be biologically relevant or actionable if based solely on a gene set, but rather on groups of coordinated gene sets. For example, while some cancer cells share the same underlying genotype, their specific phenotypes may vary. Allowing for a general genomic factor that contains more specialized states provides such a description and enables more robust genotype-phenotype mappings, which are often of interest in multi-omics studies.

We also note that allowing for a hierarchy of factors helps in resolving uncertainty in some gene sets. For example, while some cells may express T cell markers, their subtypes may be hard to distinguish. This could result in one uncertain factor per subtype. Using the hierarchy would allow cells to give a larger weight to the hierarchical factor that groups those two subtypes together.

Finally, one issue with general clustering methods is that they assume that every cluster operates on the same scale. However, biologically relevant clusters may be defined at different scales. Using a hierarchy, we can take the strongest level at which a cell is defined to guarantee a robust clustering.

One issue with such a model is its robustness to lack of hierarchy. If the gene sets are sufficient descriptions of the populations, and have no overlap, how do we interpret the hierarchy? In this case we would expect no sparsity in the hierarchy, essentially. Perhaps we can use some method to automatically find the number of hierarchical factors? Cue in Automatic Relevance Determination?x

Another issue with factorization methods is model selection, which is aggravated in its deep variations due to the increased number of layers. To choose the number of gene sets and of gene set groups, we run multiple models, evaluate their predictive performances on a held-out set of cells, and retain the best one.

Here we will use deep exponential families (DEFs) to learn a hierarchical factorization of single-cell RNA-sequencing data. In two real data sets, we show that this model is suited to simultaneously identify sparse gene sets, connections between them, and describe cells in terms of these two levels. The hierarchical level can be used to summarize general populations such as T cells, which can have specific subpopulations such as Th1 and Th2. To retrieve a general gene signature associated with the hierarchical level, we can simply take the mean of all of its children gene sets, weighted by how strong their connection to the parent is. To enable fast and practical runtimes, we develop a variational expectation-maximization scheme that identifies the number of factors automatically using automatic relevance determination. We compare our method with the standard scRNA-seq data analysis pipeline to identify gene signatures, as well as a non-deep factorization.

## PBMC data
Let's apply this method to the 10x Genomics public PBMC data set. In this data set there is a population of CD4+ T cells which contains both Naive and Memory subtypes. We can use a DEF to see this hierarchy.

## Vancouver data
This data set contains cancer cells from three copy number clones that can be identified with clonealign (Campbell et al, 2019). Our model identifies hierarchical factors that correlate with such clones, as well as shared and clone-specific cell states.

If we know the copy number clones a priori, we can bias the hierarchical factors towards their profiles, in which case our model simultaneously assigns cells to clones and finds clone-specific and clone-shared substructure.

## Melanoma cohort

For the melanoma cohort we have a spreadsheet of pathways/gene sets organised by topic, like tumor activating, hallmark, immunoregulatory, drug target, stress response and metabolic. These seem great to include as higher level topics in the DEF!

Can we make the DEF find topics which are specific to/shared by the technologies? What type of joint analysis can we do on scRNA/scDNA/CyTOF? Common latent hierarchical topic structure W, and technology+cell specific weights on the topics.

See also Satija's kNN approach, for example.

## Model selection?
Choosing the number of factors is not that important if we just make some recommendations and the results don't vary so much by changing the numbers slightly (we just need to show that they don't).

Instead of focusing on model selection, focus on the model parameterization itself.


## Modelling
The DEF has a problem of requiring a fixed number of units in each layer, which is not great to motivate to biologists. Ideally we would have an ARD-like prior on the layers in order to learn the dimensionality automatically. While there are methods to do this for a single layer (e.g. Bayesian nonparameteric Poisson factorization for recommendation systems, Gopalan 2014 or Automatic Relevance Determination in Nonnegative Matrix Factorization with the \beta-Divergence, Tan 2012), it is not clear how to do it for multiple layers. Deep Gaussian processes (Damianou 2012) provide this automatic dimensionality learning naturally, but they have the problem of being Gaussian! Can we do some approximation to Deep GPs to make them have Gamma-like layers to be usable on scRNA-seq data? Or can we generalize the single-layer methods from Gopalan 2014 or Tan 2012 into more than two layers?

# from jax import jit, vmap, grad
#
# n_cells, n_genes = data.shape
#
# n_hfactors = 10 # big number that we surely won't need
# n_factors = 30 # big number that we surely won't need
#
# cell_scale = Gamma(1, 1, shape=n_cells)
# gene_scale = Gamma(1, 1, shape=n_genes)
#
# h_factors_scale = Gamma(0.3, 1, shape=n_hfactors) # to turn off dimensions
# factors_scale = Gamma(0.3, 1, shape=n_factors) # to turn off dimensions
#
# hz = Gamma(0.3, 1. / cell_scale, shape=[n_cells, n_hfactors])
# hW = Gamma(0.3, 1. / h_factors_scale, shape=[n_hfactors, n_factors])
# z = Gamma(0.1, 0.1 / hz.dot(hW), shape=[n_cells, n_hfactors])
# W = Gamma(0.1, 0.1 / (gene_scale * factors_scale), shape=[n_hfactors, n_factors])
# x = Poisson(z.dot(W))
