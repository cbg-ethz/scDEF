if (!requireNamespace("remotes", quietly = TRUE)) {
  install.packages("remotes", repos = "https://cran.r-project.org")
}
if (!requireNamespace("SeuratData", quietly = TRUE)) {
  remotes::install_github("satijalab/seurat-data", upgrade = "never", quiet = TRUE)
}
options(timeout = 600)
suppressMessages(library(Seurat))
suppressMessages(library(SeuratData))

dataset <- snakemake@params[["dataset"]]
package <- tryCatch(snakemake@params[["package"]], error = function(e) NULL)
if (is.null(package)) package <- dataset
counts_fname <- snakemake@output[["counts_fname"]]
meta_fname <- snakemake@output[["meta_fname"]]

cat("Downloading SeuratData package:", package, " dataset:", dataset, "\n")

if (!requireNamespace(paste0(package, ".SeuratData"), quietly = TRUE)) {
  InstallData(package)
}

data(list = dataset, package = paste0(package, ".SeuratData"))
obj <- get(dataset)

# Update old Seurat v3 objects to v5 format
obj <- tryCatch(UpdateSeuratObject(obj), error = function(e) obj)

counts <- GetAssayData(object = obj, layer = "counts")
meta <- obj@meta.data

dir.create(dirname(counts_fname), showWarnings = FALSE, recursive = TRUE)

cat("Writing counts matrix:", dim(counts)[1], "genes x", dim(counts)[2], "cells\n")

# For large matrices, save in Matrix Market format to avoid memory issues
if (prod(dim(counts)) > 1e8) {
  library(Matrix)
  mtx_path <- sub("\\.csv$", ".mtx", counts_fname)
  genes_path <- sub("\\.csv$", "_genes.tsv", counts_fname)
  barcodes_path <- sub("\\.csv$", "_barcodes.tsv", counts_fname)
  Matrix::writeMM(counts, file = mtx_path)
  writeLines(rownames(counts), genes_path)
  writeLines(colnames(counts), barcodes_path)
  # Write a sentinel so the Python script knows to read MTX
  writeLines("__MTX_FORMAT__", counts_fname)
  cat("Saved as Matrix Market (sparse) format\n")
} else {
  write.csv(as.data.frame(as.matrix(counts)), file = counts_fname)
}
write.csv(meta, file = meta_fname)

cat("Done.\n")
