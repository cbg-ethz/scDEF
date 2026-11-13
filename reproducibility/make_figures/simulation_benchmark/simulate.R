# The 2024 simulation generator (reproducibility/workflow/scripts/splatter_hierarchical.R), copied for the
# simulation benchmark re-run (scDEF-markdown/experiments/exp-simulation-benchmark-rerun.md). Every change
# from 2024 is marked "ADDED (2026-11-05)"; the rest is the 2024 code. The changes:
#   1. a command-line front end that builds the `snakemake` object the 2024 code reads;
#   2. the pbmc3k reference for splatEstimate comes from a counts file when SeuratData is not installed
#      (it is not on conda): scanpy's copy of the same 10x data, filtered as SeuratData's pbmc3k
#      (genes in >= 3 cells, cells with >= 200 genes: 13,714 x 2,700); the deterministic estimate is cached;
#   3. splatter's ground truth is saved before the rbind: every gene's DE factor for every group of its block;
#   4. set.seed(seed) before the batch-specific groups are drawn: splatter 1.26 (used here) runs each simulation inside
#      withr::with_seed, so without it that draw is unseeded and the data set is not reproducible;
#   5. the UMAP PNGs are optional (umap=TRUE), off by default.
#   6. the generator's shared and batch-specific groups are saved (groups_fname), for the shared/specific metric.
#
#   Rscript simulate.R n_cells=4000 n_batches=1 frac_shared=0 de_fscale=2 de_prob=0.2 batch_facscale=0 \
#     coverage=0 seed=0 counts_fname=... meta_fname=... markers_fname=... defacs_fname=... \
#     reference=pbmc3k_counts.mtx params_rds=splat_params.rds [umap=FALSE umap_fname=... umap_nobatch_fname=...]

# ---- ADDED (2026-11-05): command-line front end in place of Snakemake --------------------------------------
.args <- commandArgs(trailingOnly = TRUE)
.kv <- setNames(sub("^[^=]*=", "", .args), sub("=.*$", "", .args))
.get <- function(k, default = NULL) if (k %in% names(.kv)) .kv[[k]] else default
setClass("SnakemakeShim", representation(params = "list", output = "list"))
snakemake <- new("SnakemakeShim",
  params = list(n_cells = .get("n_cells"), n_batches = .get("n_batches"), frac_shared = .get("frac_shared"),
                de_fscale = .get("de_fscale"), de_prob = .get("de_prob"),
                batch_facscale = .get("batch_facscale"), coverage = .get("coverage", "0"),
                seed = .get("seed")),
  output = list(counts_fname = .get("counts_fname"), meta_fname = .get("meta_fname"),
                markers_fname = .get("markers_fname"), umap_fname = .get("umap_fname"),
                umap_nobatch_fname = .get("umap_nobatch_fname")))
defacs_fname <- .get("defacs_fname")
reference_mtx <- .get("reference")
params_rds <- .get("params_rds")
make_umap <- as.logical(.get("umap", "FALSE"))
groups_fname <- .get("groups_fname")                        # ADDED (2026-11-05, later the same day): see 6.
use_seuratdata <- requireNamespace("SeuratData", quietly = TRUE) &&
  "pbmc3k.SeuratData" %in% rownames(installed.packages())
# ------------------------------------------------------------------------------------------------------------

suppressMessages(library(splatter))
suppressMessages(library(ggplot2))
suppressMessages(library(cowplot))
suppressMessages(library(SingleCellExperiment))
if (use_seuratdata) suppressMessages(library(SeuratData))   # ADDED (2026-11-05): the condition
suppressMessages(library(Seurat))
suppressMessages(library(dplyr))


n_groups <- 8
n_cells_per_batch <- as.integer(snakemake@params[["n_cells"]])
n_batches <- as.integer(snakemake@params[["n_batches"]])
n_cells <- rep(n_cells_per_batch, n_batches)
n_total_cells <- as.integer(n_cells_per_batch) * as.integer(n_batches)
frac_shared <- as.numeric(snakemake@params[["frac_shared"]])
de_fscale <- as.numeric(snakemake@params[["de_fscale"]])
de_prob <- as.numeric(snakemake@params[["de_prob"]])
batch_facscale <- as.numeric(snakemake@params[["batch_facscale"]])
coverage <- as.numeric(snakemake@params[["coverage"]])
seed <- as.numeric(snakemake@params[["seed"]])


if (!is.null(params_rds) && file.exists(params_rds)) {   # ADDED (2026-11-05): cached estimate
  params <- readRDS(params_rds)
} else {
if (use_seuratdata) {                                       # ADDED (2026-11-05): the condition
# Get 3k PBMCs
# InstallData("pbmc3k")
data("pbmc3k")
c <- as.matrix(GetAssayData(object = pbmc3k, slot = "counts")) # Use 3k PBMCs data from 10x Genomics as reference
} else {                                                    # ADDED (2026-11-05): scanpy's copy of the same data
c <- as.matrix(Matrix::readMM(reference_mtx))
}
params <- splatEstimate(c)
if (!is.null(params_rds)) {                                 # ADDED (2026-11-05): cache (written atomically)
  saveRDS(params, paste0(params_rds, ".tmp"))
  file.rename(paste0(params_rds, ".tmp"), params_rds)
}
}

print(params)

# Generate 4000 cells with 1000 genes separated in 2 groups
params <- setParam(params, "nGenes", 1000)
params <- setParam(params, "batchCells", n_cells)
params <- setParam(params, "de.prob", de_prob)
params <- setParam(params, "de.facScale", de_fscale)
params <- setParam(params, "seed", seed)
if (coverage > 0)
  params <- setParam(params, "lib.loc", log(coverage * 1000/1750))
params <- setParam(params, "group.prob", rep(1/2, 2))
sim1 <- splatSimulate(params, method="groups", verbose=FALSE)
# sort cells by groups
# sim1 <- sim1[,order(colData(sim)$Group)]
# rename cells
colData(sim1)$Cell <- paste0("Cell", seq(n_total_cells))
colnames(sim1) <- colData(sim1)$Cell
# name group column according to hierarchy level
colData(sim1)$GroupA <- colData(sim1)$Group
colData(sim1) <- subset(colData(sim1), select = -c(Group, ExpLibSize))
print("First splatter done!")

# Add 500 genes separated in 4 groups
params <- setParam(params, "nGenes", 500)
params <- setParam(params, "batchCells", n_cells)
params <- setParam(params, "de.prob", de_prob)
params <- setParam(params, "de.facScale", de_fscale)
params <- setParam(params, "seed", seed)
if (coverage > 0)
  params <- setParam(params, "lib.loc", log(coverage * 500/1750))
params <- setParam(params, "group.prob", rep(1/4, 4))
sim2 <- splatSimulate(params, method="groups", verbose=FALSE)
# # sort cells by groups
# sim2 <- sim2[,order(colData(sim2)$Group)]
# rename cells
colData(sim2)$Cell <- paste0("Cell", seq(n_total_cells))
colnames(sim2) <- colData(sim2)$Cell
# name group column according to hierarchy level
colData(sim2)$GroupB <- colData(sim2)$Group
colData(sim2) <- subset(colData(sim2), select = -c(Group, ExpLibSize))
print("Second splatter done!")

# Add 250 genes separated in 8 groups
params <- setParam(params, "nGenes", 250)
params <- setParam(params, "batchCells", n_cells)
params <- setParam(params, "de.prob", de_prob)
params <- setParam(params, "de.facScale", de_fscale)
params <- setParam(params, "seed", seed)
params <- setParam(params, "group.prob", rep(1/8, 8))
if (coverage > 0)
  params <- setParam(params, "lib.loc", log(coverage * 250/1750))
params <- setParam(params, "batch.facScale", batch_facscale)
sim3 <- splatSimulate(params, method="groups", verbose=FALSE)
sim3_nobatch <- splatSimulate(params, method="groups", verbose=FALSE, batch.rmEffect = TRUE,)
# sort cells by groups
# sim3 <- sim3[,order(colData(sim3)$Group)]
# sim3_nobatch <- sim3_nobatch[,order(colData(sim3_nobatch)$Group)]
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

# ---- ADDED (2026-11-05): splatter's ground truth, saved before the rbind (which unions the DEFacGroup<k>
# columns of the three blocks into one table). One row per gene and group of its block: the final gene name
# (Gene1-1000 block A, Gene1001-1500 block B, Gene1501-1750 block C, as the rename below), the level whose
# labels the group uses (GroupA/GroupB/GroupC), the group label as in that column, and splatter's DE factor
# (> 1 up-regulated in the group, < 1 down, 1 not DE). Block C comes from `sim3` (the factors of
# `sim3_nobatch` are checked to be the same).
defac_rows <- function(s, level, offset) {
  rd <- as.data.frame(rowData(s))
  cols <- grep("^DEFacGroup", colnames(rd), value = TRUE)
  do.call(rbind, lapply(cols, function(cl) data.frame(
    gene = paste0("Gene", offset + seq_len(nrow(rd))), level = level,
    group = sub("^DEFac", "", cl), de_factor = rd[[cl]], gene_mean = rd$GeneMean)))
}
.d3 <- as.data.frame(rowData(sim3))[grep("^DEFacGroup", colnames(rowData(sim3)))]
.d3n <- as.data.frame(rowData(sim3_nobatch))[grep("^DEFacGroup", colnames(rowData(sim3_nobatch)))]
if (!isTRUE(all.equal(.d3, .d3n))) stop("DE factors of sim3 and sim3_nobatch differ")
defacs <- rbind(defac_rows(sim1, "GroupA", 0), defac_rows(sim2, "GroupB", nrow(sim1)),
                defac_rows(sim3, "GroupC", nrow(sim1) + nrow(sim2)))
write.csv(defacs, defacs_fname, row.names = FALSE)
# ------------------------------------------------------------------------------------------------------------

# Append all genes
sim_nobatch <- rbind(sim1,sim2,sim3_nobatch)
sim <- rbind(sim1,sim2,sim3)
rownames(sim) <- paste0("Gene", seq(1750))
rownames(sim_nobatch) <- paste0("Gene", seq(1750))
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
# sim <- sim[,colData(sim)$Group == possibleGroups]
# sim_nobatch <- sim_nobatch[,colData(sim_nobatch)$Group == possibleGroups]
# Sort groups
sim <- sim[,order(colData(sim)$Group)]
sim_nobatch <- sim_nobatch[,order(colData(sim_nobatch)$Group)]

print("Sim done!")

if (n_batches > 1) {
  set.seed(seed)   # ADDED (2026-11-05): after splatSimulate (withr::with_seed) the global RNG is unseeded; makes the draw below reproducible
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
  # ---- ADDED (2026-11-05): the generator's shared groups (`shared`, kept in every batch) and batch-specific
  # groups (`tokeep_per_batch`, one batch each), with the GroupC label of each group.
  if (!is.null(groups_fname)) {
    gmap <- unique(data.frame(Group = sim$Group, GroupC = sim$GroupC))
    sh <- if (n_shared > 0) shared else character(0)
    grows <- data.frame(Group = sh, kind = rep("shared", length(sh)),
                        batches = rep(paste(sort(unique(sim$Batch)), collapse = ";"), length(sh)))
    if (n_shared < n_groups) {
      for (b in names(tokeep_per_batch)) for (g in unlist(tokeep_per_batch[[b]]))
        grows <- rbind(grows, data.frame(Group = g, kind = "specific", batches = b))
    }
    grows$GroupC <- gmap$GroupC[match(grows$Group, gmap$Group)]
    write.csv(grows, groups_fname, row.names = FALSE)
  }
  # ------------------------------------------------------------------------------------------------------------
  # Make Seurat object
  # sub_sim <- logNormCounts(sub_sim)
  # sub_sim_nobatch <- logNormCounts(sub_sim_nobatch)
  sub_sim.seurat <- as.Seurat(sub_sim, counts = "counts", data=NULL)
  sub_sim.seurat <- NormalizeData(sub_sim.seurat, verbose = FALSE)
  sub_sim_nobatch.seurat <- as.Seurat(sub_sim_nobatch, counts = "counts", data=NULL)
  sub_sim_nobatch.seurat <- NormalizeData(sub_sim_nobatch.seurat, verbose = FALSE)

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

  if (make_umap) {   # ADDED (2026-11-05): the condition
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

  # sub_sim_nobatch.seurat <- NormalizeData(sub_sim_nobatch.seurat, verbose = FALSE)
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
  }   # ADDED (2026-11-05): end of the condition

  print("Batch subsets done!")
} else {
  sub_sim <- sim
  if (!is.null(groups_fname)) {   # ADDED (2026-11-05): one batch, every group in it
    gmap <- unique(data.frame(Group = sim$Group, GroupC = sim$GroupC))
    write.csv(data.frame(Group = gmap$Group, kind = "one batch", batches = paste(unique(sim$Batch), collapse = ";"),
                         GroupC = gmap$GroupC), groups_fname, row.names = FALSE)
  }

  # Make Seurat object
  # sub_sim <- logNormCounts(sub_sim)
  sub_sim.seurat <- as.Seurat(sub_sim, counts = "counts", data=NULL)
  sub_sim.seurat <- NormalizeData(sub_sim.seurat, verbose = FALSE)

  # Save group signatures from non-batch data
  Idents(object = sub_sim.seurat) <- "GroupC"
  sub_sim.seurat.markers <- FindAllMarkers(sub_sim.seurat, only.pos = TRUE, min.pct = 0.25, logfc.threshold = 0.25)
  markers <- sub_sim.seurat.markers %>%
      group_by(cluster) %>%
      slice_max(n = 10, order_by = avg_log2FC) %>%
      filter(p_val_adj < 0.05)
  write.csv(markers,snakemake@output[["markers_fname"]])

  # Save the data
  counts = data.frame(counts(sub_sim))
  meta = data.frame(colData(sub_sim))
  write.csv(counts,snakemake@output[["counts_fname"]])
  write.csv(meta,snakemake@output[["meta_fname"]])

  if (make_umap) {   # ADDED (2026-11-05): the condition
  # Save the UMAP of the data
  # sub_sim.seurat <- NormalizeData(sub_sim.seurat, verbose = FALSE)
  sub_sim.seurat <- FindVariableFeatures(sub_sim.seurat, selection.method = "vst", nfeatures = 1000)
  sub_sim.seurat <- ScaleData(sub_sim.seurat, verbose = FALSE)
  sub_sim.seurat <- RunPCA(sub_sim.seurat, npcs = 30, verbose = FALSE)
  sub_sim.seurat <- RunUMAP(sub_sim.seurat, reduction = "pca", dims = 1:20)
  sub_sim.seurat <- FindNeighbors(sub_sim.seurat, reduction = "pca", dims = 1:20)
  sub_sim.seurat <- FindClusters(sub_sim.seurat, resolution = 0.5)
  p1 <- DimPlot(sub_sim.seurat, reduction = "umap", group.by = "GroupA", label = TRUE)
  p2 <- DimPlot(sub_sim.seurat, reduction = "umap", group.by = "GroupB", label = TRUE)
  p3 <- DimPlot(sub_sim.seurat, reduction = "umap", group.by = "GroupC", label = TRUE)
  p4 <- DimPlot(sub_sim.seurat, reduction = "umap", group.by = "Group", label = TRUE)
  p5 <- DimPlot(sub_sim.seurat, reduction = "umap", label = TRUE)
  p <- plot_grid(p1, p2, p3, p4, p5, ncol = 3)
  ggsave(snakemake@output[["umap_fname"]])
  }   # ADDED (2026-11-05): end of the condition
}
