suppressMessages(library(splatter))
suppressMessages(library(ggplot2))
suppressMessages(library(scater))


params <- newSplatParams()
params <- setParam(params, "nGenes", snakemake@params[["n_genes"]])
params <- setParam(params, "batchCells", c(snakemake@params[["n_cells"]]))
params <- setParam(params, "de.prob", snakemake@params[["de_prob"]])
params <- setParam(params, "de.facScale", as.numeric(snakemake@params[["de_facscale"]]))
params <- setParam(params, "seed", as.numeric(snakemake@params[["seed"]]))
group_probs <- rep(1/snakemake@params[["n_groups"]], snakemake@params[["n_groups"]])
params <- setParam(params, "group.prob", group_probs)

sim <- splatSimulate(params, method="groups", verbose=FALSE)

counts = data.frame(counts(sim))
meta = data.frame(colData(sim))

sim <- logNormCounts(sim)
sim <- runPCA(sim)
g <- plotPCA(sim, colour_by = "Group")

# Save the plot
ggsave(snakemake@output[["plot_fname"]])

# Save the data
write.csv(counts,snakemake@output[["counts_fname"]])
write.csv(meta,snakemake@output[["meta_fname"]])
