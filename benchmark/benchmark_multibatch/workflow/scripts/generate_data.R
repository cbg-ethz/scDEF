suppressMessages(library(splatter))
suppressMessages(library(ggplot2))
suppressMessages(library(scater))
suppressMessages(library(SeuratData))
suppressMessages(library(Seurat))
suppressMessages(library(dplyr))

n_cells_per_batch <- snakemake@params[["n_cells"]]
n_batches <- snakemake@params[["n_batches"]]
n_cells <- rep(n_cells_per_batch, n_batches)
frac_shared <- as.numeric(snakemake@params[["frac_shared"]])

# Shared groups
n_groups <- as.numeric(snakemake@params[["n_groups"]])

# Create a Splat with all batches
data("pbmc3k")
c <- as.matrix(GetAssayData(object = pbmc3k, slot = "counts"))

params <- splatEstimate(c)
params <- setParam(params, "nGenes", snakemake@params[["n_genes"]])
params <- setParam(params, "batchCells", n_cells)
params <- setParam(params, "de.prob", as.numeric(snakemake@params[["de_prob"]]))
params <- setParam(params, "batch.facScale", as.numeric(snakemake@params[["batch_facscale"]]))
params <- setParam(params, "seed", as.numeric(snakemake@params[["seed"]]))
group_probs <- rep(1/n_groups, n_groups)
params <- setParam(params, "group.prob", group_probs)

sim <- splatSimulate(params, method="groups", verbose=FALSE)

group_list <- as.character(as.list(unique(sim$Group)))
group_list <- paste0("Group", group_list)

# Select groups to be shared
n_shared <- round(frac_shared*n_groups)
if (n_shared > 0)
  shared <- group_list[1:n_shared]
specific <- group_list[(n_shared+1):n_groups]

# Now sample non-shared groups on each batch
sub_sim <- sim[,sim$Batch == ""]
batches <- unique(sim$Batch)
tokeep_per_batch <- setNames(rep(list(list()), length(batches)), batches)

while(length(specific) > 0) {
  for (b in batches) {
    if (length(specific) > 0) {
      # Select specific groups to keep
      batch_groups <- sample(specific, 1, replace=FALSE)
      specific <- specific[!specific %in% batch_groups]
      tokeep_per_batch[[b]] <- append(tokeep_per_batch[[b]], batch_groups)
    }
  }
}

for (b in batches) {
  # Get batch
  if (n_shared > 0) {
    tokeep <- c(shared, unlist(tokeep_per_batch[[b]]))
  } else {
    tokeep <- tokeep_per_batch[[b]]
  }
  print(b)
  print(tokeep)
  batch_sce <- sim[,sim$Batch == b]
  batch_sce <- batch_sce[,batch_sce$Group %in% tokeep]
  # Append to new SCE
  sub_sim <- cbind(sub_sim, batch_sce)
}

counts = data.frame(counts(sub_sim))
meta = data.frame(colData(sub_sim))

sub_sim <- logNormCounts(sub_sim)
sub_sim <- runPCA(sub_sim)

# Save the plots
g <- plotPCA(sub_sim, colour_by = "Group")
ggsave(snakemake@output[["plot_groups_fname"]])

g <- plotPCA(sub_sim, colour_by = "Batch")
ggsave(snakemake@output[["plot_batches_fname"]])

# Get gene signatures for each group
sub_sim.seurat <- as.Seurat(sub_sim, counts = "counts", data = "logcounts")

Idents(object = sub_sim.seurat) <- "Group"
sub_sim.seurat.markers <- FindAllMarkers(sub_sim.seurat, only.pos = TRUE, min.pct = 0.25, logfc.threshold = 0.25)
markers <- sub_sim.seurat.markers %>%
    group_by(cluster) %>%
    slice_max(n = 5, order_by = avg_log2FC) %>%
    filter(p_val_adj < 0.05)

# Save the data
write.csv(counts,snakemake@output[["counts_fname"]])
write.csv(meta,snakemake@output[["meta_fname"]])
write.csv(markers,snakemake@output[["markers_fname"]])
