import anndata
import pandas as pd
import scipy.sparse as sp
import scipy.io


def main():
    counts_fname = snakemake.input["counts_fname"]
    meta = pd.read_csv(snakemake.input["meta_fname"], index_col=0)

    # Check if R saved in Matrix Market format (for large datasets)
    with open(counts_fname) as f:
        first_line = f.readline().strip()

    if first_line == "__MTX_FORMAT__":
        mtx_path = counts_fname.replace(".csv", ".mtx")
        genes_path = counts_fname.replace(".csv", "_genes.tsv")
        barcodes_path = counts_fname.replace(".csv", "_barcodes.tsv")
        print(f"Reading Matrix Market format from {mtx_path}")
        X = scipy.io.mmread(mtx_path).T.tocsr()  # genes x cells → cells x genes
        with open(genes_path) as f:
            genes = [line.strip() for line in f]
        with open(barcodes_path) as f:
            barcodes = [line.strip() for line in f]
        meta = meta.loc[barcodes]
        adata = anndata.AnnData(
            X=X,
            obs=meta,
            var=pd.DataFrame(index=genes),
        )
    else:
        counts = pd.read_csv(counts_fname, index_col=0)
        # Ensure cell barcode alignment
        common_cells = counts.columns.intersection(meta.index)
        if len(common_cells) < len(meta.index):
            print(f"Warning: {len(meta.index) - len(common_cells)} cells in metadata not found in counts")
        counts = counts[common_cells]
        meta = meta.loc[common_cells]
        adata = anndata.AnnData(
            X=sp.csr_matrix(counts.values.T),
            obs=meta,
            var=pd.DataFrame(index=counts.index),
        )

    adata.write_h5ad(snakemake.output["fname"])
    print(f"Saved {adata.shape[0]} cells x {adata.shape[1]} genes to {snakemake.output['fname']}")


if __name__ == "__main__":
    main()
