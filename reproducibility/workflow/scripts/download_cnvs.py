"""Download SPECTRUM HGSOC atlas from CellxGene and gene annotations from Ensembl BioMart."""
import os
import urllib.request
import json

CELLXGENE_ID = "680499cc-f9c5-44d5-b4f8-4cdf6c12887e"
CELLXGENE_URL = f"https://datasets.cellxgene.cziscience.com/{CELLXGENE_ID}.h5ad"

BIOMART_URL = (
    "http://www.ensembl.org/biomart/martservice?"
    "query=%3C%3Fxml%20version%3D%221.0%22%20encoding%3D%22UTF-8%22%3F%3E"
    "%3C!DOCTYPE%20Query%3E"
    "%3CQuery%20virtualSchemaName%3D%22default%22%20formatter%3D%22CSV%22"
    "%20header%3D%221%22%20uniqueRows%3D%221%22%20count%3D%22%22%3E"
    "%3CDataset%20name%3D%22hsapiens_gene_ensembl%22%3E"
    "%3CAttribute%20name%3D%22hgnc_symbol%22%2F%3E"
    "%3CAttribute%20name%3D%22start_position%22%2F%3E"
    "%3CAttribute%20name%3D%22end_position%22%2F%3E"
    "%3CAttribute%20name%3D%22chromosome_name%22%2F%3E"
    "%3C%2FDataset%3E"
    "%3C%2FQuery%3E"
)


def main():
    out_dir = snakemake.params["out_dir"]
    os.makedirs(out_dir, exist_ok=True)

    h5ad_path = os.path.join(out_dir, "all_cells.h5ad")
    annot_path = os.path.join(out_dir, "annot.csv")

    print(f"Downloading CellxGene dataset {CELLXGENE_ID} ...")
    urllib.request.urlretrieve(CELLXGENE_URL, h5ad_path)
    print(f"  -> {h5ad_path}")

    print("Downloading gene annotations from Ensembl BioMart ...")
    biomart_raw = os.path.join(out_dir, "biomart_raw.csv")
    urllib.request.urlretrieve(BIOMART_URL, biomart_raw)

    import pandas as pd
    df = pd.read_csv(biomart_raw)
    df.columns = ["hgnc_symbol", "start_position", "end_position", "chromosome_name"]
    df = df[df["hgnc_symbol"].notna() & (df["hgnc_symbol"] != "")]
    df = df.drop_duplicates(subset="hgnc_symbol", keep="first")
    df = df.set_index("hgnc_symbol")
    df.to_csv(annot_path)
    os.remove(biomart_raw)
    print(f"  -> {annot_path} ({len(df)} genes)")

    with open(snakemake.output["done"], "w") as f:
        f.write("done\n")

    print("Done.")


if __name__ == "__main__":
    main()
