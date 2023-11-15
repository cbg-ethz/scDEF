suppressMessages(library(splatter))
suppressMessages(library(ggplot2))
suppressMessages(library(cowplot))
suppressMessages(library(scater))
suppressMessages(library(SeuratData))
suppressMessages(library(Seurat))
suppressMessages(library(dplyr))


n_groups <- 8
n_cells_per_batch <- snakemake@params[["n_cells"]]
n_batches <- snakemake@params[["n_batches"]]
n_cells <- rep(n_cells_per_batch, n_batches)
n_total_cells <- as.integer(n_cells_per_batch) * as.integer(n_batches)
frac_shared <- as.numeric(snakemake@params[["frac_shared"]])
de_prob <- as.numeric(snakemake@params[["de_prob"]])
batch_facscale <- as.numeric(snakemake@params[["batch_facscale"]])
seed <- as.numeric(snakemake@params[["seed"]])


# Get 3k PBMCs
data("pbmc3k")
c <- as.matrix(GetAssayData(object = pbmc3k, slot = "counts")) # Use 3k PBMCs data from 10x Genomics as reference
params <- splatEstimate(c)

# Generate 4000 cells with 1000 genes separated in 2 groups
params <- setParam(params, "nGenes", 1000)
params <- setParam(params, "batchCells", n_total_cells)
params <- setParam(params, "de.prob", de_prob)
params <- setParam(params, "seed", seed)
params <- setParam(params, "group.prob", rep(1/2, 2))
sim <- splatSimulate(params, method="groups", verbose=FALSE)
# sort cells by groups
sim <- sim[,order(colData(sim)$Group)]
# rename cells
colData(sim)$Cell <- paste0("Cell", seq(n_total_cells))
colnames(sim) <- colData(sim)$Cell
# name group column according to hierarchy level
colData(sim)$GroupA <- colData(sim)$Group
colData(sim) <- subset(colData(sim), select = -c(Group, ExpLibSize, Batch))
print("First splatter done!")

# Add 500 genes separated in 4 groups
params <- setParam(params, "nGenes", 500)
params <- setParam(params, "batchCells", n_total_cells)
params <- setParam(params, "de.prob", de_prob)
params <- setParam(params, "seed", seed)
params <- setParam(params, "group.prob", rep(1/4, 4))
sim2 <- splatSimulate(params, method="groups", verbose=FALSE)
# sort cells by groups
sim2 <- sim2[,order(colData(sim2)$Group)]
# rename cells
colData(sim2)$Cell <- paste0("Cell", seq(n_total_cells))
colnames(sim2) <- colData(sim2)$Cell
# name group column according to hierarchy level
colData(sim2)$GroupB <- colData(sim2)$Group
colData(sim2) <- subset(colData(sim2), select = -c(Group, ExpLibSize, Batch))
print("Second splatter done!")

# Add 250 genes separated in 8 groups
params <- setParam(params, "nGenes", 500)
params <- setParam(params, "batchCells", n_cells)
params <- setParam(params, "de.prob", de_prob)
params <- setParam(params, "seed", seed)
params <- setParam(params, "group.prob", rep(1/8, 8))
params <- setParam(params, "batch.facScale", batch_facscale)
sim3 <- splatSimulate(params, method="groups", verbose=FALSE)
sim3_nobatch <- splatSimulate(params, method="groups", verbose=FALSE, batch.rmEffect = TRUE,)
# sort cells by groups
sim3 <- sim3[,order(colData(sim3)$Group)]
sim3_nobatch <- sim3_nobatch[,order(colData(sim3_nobatch)$Group)]
# rename cells
colData(sim3)$Cell <- paste0("Cell", seq(n_total_cells))
colData(sim3_nobatch)$Cell <- paste0("Cell", seq(n_total_cells))
colnames(sim3) <- colData(sim3)$Cell
colnames(sim3_nobatch) <- colData(sim3_nobatch)$Cell
# name group column according to hierarchy level
colData(sim3)$GroupC <- colData(sim3)$Group
colData(sim3_nobatch)$GroupC <- colData(sim3_nobatch)$Group
colData(sim3) <- subset(colData(sim3), select = -c(Group,ExpLibSize))
colData(sim3_nobatch) <- subset(colData(sim3_nobatch), select = -c(Group,ExpLibSize))
print("Third splatter done!")

# Append all genes
sim_nobatch <- rbind(sim,sim2,sim3_nobatch)
sim <- rbind(sim,sim2,sim3)
rownames(sim) <- paste0("Gene", seq(2000))
rownames(sim_nobatch) <- paste0("Gene", seq(2000))
# Make groups
colData(sim)$Group <- paste0(colData(sim)$GroupA,colData(sim)$GroupB,colData(sim)$GroupC)
colData(sim_nobatch)$Group <- paste0(colData(sim_nobatch)$GroupA,colData(sim_nobatch)$GroupB,colData(sim_nobatch)$GroupC)
# Drop cells with inconsistent groups
possibleGroups <- c(rep("Group1Group1Group1", n_total_cells/8),
                    rep("Group1Group1Group2", n_total_cells/8),
                    rep("Group1Group2Group3", n_total_cells/8),
                    rep("Group1Group2Group4", n_total_cells/8),
                    rep("Group2Group3Group5", n_total_cells/8),
                    rep("Group2Group3Group6", n_total_cells/8),
                    rep("Group2Group4Group7", n_total_cells/8),
                    rep("Group2Group4Group8", n_total_cells/8)
                    )
sim <- sim[,colData(sim)$Group == possibleGroups]
sim_nobatch <- sim_nobatch[,colData(sim_nobatch)$Group == possibleGroups]
print("Sim done!")
# Drop some cell groups from some batches
group_list <- as.character(as.list(unique(sim$Group)))
n_shared <- round(frac_shared*n_groups)
if (n_shared > 0)
  shared <- group_list[1:n_shared]
specific <- group_list[(n_shared+1):n_groups]
if (n_shared < n_groups) {
  # Now sample non-shared groups on each batch
  sub_sim <- sim[,sim$Batch == ""]
  sub_sim_nobatch <- sim_nobatch[,sim_nobatch$Batch == ""]
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
    batch_sce <- sim[,sim$Batch == b]
    batch_sce <- batch_sce[,batch_sce$Group %in% tokeep]

    batch_sce_nobatch <- sim_nobatch[,sim_nobatch$Batch == b]
    batch_sce_nobatch <- batch_sce_nobatch[,batch_sce_nobatch$Group %in% tokeep]

    # Append to new SCE
    sub_sim <- cbind(sub_sim, batch_sce)
    sub_sim_nobatch <- cbind(sub_sim_nobatch, batch_sce_nobatch)
  }
} else {
  sub_sim <- sim
  sub_sim_nobatch <- sim_nobatch
}
print("Batch subsets done!")

# Make Seurat object
sub_sim <- logNormCounts(sub_sim)
sub_sim_nobatch <- logNormCounts(sub_sim_nobatch)
sub_sim.seurat <- as.Seurat(sub_sim, counts = "counts", logcounts = "logcounts")
sub_sim_nobatch.seurat <- as.Seurat(sub_sim_nobatch, counts = "counts", logcounts = "logcounts")

# Save group signatures from non-batch data
Idents(object = sub_sim_nobatch.seurat) <- "GroupC"
sub_sim_nobatch.seurat.markers <- FindAllMarkers(sub_sim_nobatch.seurat, only.pos = TRUE, min.pct = 0.25, logfc.threshold = 0.25)
markers <- sub_sim_nobatch.seurat.markers %>%
    group_by(cluster) %>%
    slice_max(n = 10, order_by = avg_log2FC) %>%
    filter(p_val_adj < 0.05)
write.csv(markers,snakemake@output[["markers_fname"]])

# Save the data
counts = data.frame(counts(sub_sim))
meta = data.frame(colData(sub_sim))
write.csv(counts,snakemake@output[["counts_fname"]])
write.csv(meta,snakemake@output[["meta_fname"]])

# Save the UMAP of the data
sub_sim.seurat <- NormalizeData(sub_sim.seurat, verbose = FALSE)
sub_sim.seurat <- FindVariableFeatures(sub_sim.seurat, selection.method = "vst", nfeatures = 1000)
sub_sim.seurat <- ScaleData(sub_sim.seurat, verbose = FALSE)
sub_sim.seurat <- RunPCA(sub_sim.seurat, npcs = 30, verbose = FALSE)
sub_sim.seurat <- RunUMAP(sub_sim.seurat, reduction = "pca", dims = 1:20)
sub_sim.seurat <- FindNeighbors(sub_sim.seurat, reduction = "pca", dims = 1:20)
sub_sim.seurat <- FindClusters(sub_sim.seurat, resolution = 0.5)
p1 <- DimPlot(sub_sim.seurat, reduction = "umap", group.by = "GroupA", label = TRUE)
p2 <- DimPlot(sub_sim.seurat, reduction = "umap", group.by = "GroupB", label = TRUE)
p3 <- DimPlot(sub_sim.seurat, reduction = "umap", group.by = "GroupC", label = TRUE)
p4 <- DimPlot(sub_sim.seurat, reduction = "umap", group.by = "Batch", label = TRUE)
p5 <- DimPlot(sub_sim.seurat, reduction = "umap", group.by = "Group", label = TRUE)
p6 <- DimPlot(sub_sim.seurat, reduction = "umap", label = TRUE)
p <- plot_grid(p1, p2, p3, p4, p5, p6, ncol = 3)
ggsave(snakemake@output[["umap_fname"]])

sub_sim_nobatch.seurat <- NormalizeData(sub_sim_nobatch.seurat, verbose = FALSE)
sub_sim_nobatch.seurat <- FindVariableFeatures(sub_sim_nobatch.seurat, selection.method = "vst", nfeatures = 1000)
sub_sim_nobatch.seurat <- ScaleData(sub_sim_nobatch.seurat, verbose = FALSE)
sub_sim_nobatch.seurat <- RunPCA(sub_sim_nobatch.seurat, npcs = 30, verbose = FALSE)
sub_sim_nobatch.seurat <- RunUMAP(sub_sim_nobatch.seurat, reduction = "pca", dims = 1:20)
sub_sim_nobatch.seurat <- FindNeighbors(sub_sim_nobatch.seurat, reduction = "pca", dims = 1:20)
sub_sim_nobatch.seurat <- FindClusters(sub_sim_nobatch.seurat, resolution = 0.5)
p1 <- DimPlot(sub_sim_nobatch.seurat, reduction = "umap", group.by = "GroupA", label = TRUE)
p2 <- DimPlot(sub_sim_nobatch.seurat, reduction = "umap", group.by = "GroupB", label = TRUE)
p3 <- DimPlot(sub_sim_nobatch.seurat, reduction = "umap", group.by = "GroupC", label = TRUE)
p4 <- DimPlot(sub_sim_nobatch.seurat, reduction = "umap", group.by = "Batch", label = TRUE)
p5 <- DimPlot(sub_sim_nobatch.seurat, reduction = "umap", group.by = "Group", label = TRUE)
p6 <- DimPlot(sub_sim_nobatch.seurat, reduction = "umap", label = TRUE)
p <- plot_grid(p1, p2, p3, p4, p5, p6, ncol = 3)
ggsave(snakemake@output[["umap_nobatch_fname"]])
