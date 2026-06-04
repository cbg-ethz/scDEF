"""Download HGSOC chemotherapy data from GEO (GSE165897)."""
import os
import urllib.request
import gzip
import shutil

GEO_BASE = "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE165nnn/GSE165897/suppl"
FILES = {
    "GSE165897_UMIcounts_HGSOC.tsv.gz": "counts.tsv",
    "GSE165897_cellInfo_HGSOC.tsv.gz": "cellinfo.tsv",
}


def main():
    out_dir = os.path.dirname(snakemake.output["counts_fname"])
    os.makedirs(out_dir, exist_ok=True)

    for gz_name, out_name in FILES.items():
        url = f"{GEO_BASE}/{gz_name}"
        gz_path = os.path.join(out_dir, gz_name)
        out_path = os.path.join(out_dir, out_name)
        print(f"Downloading {url} ...")
        urllib.request.urlretrieve(url, gz_path)
        with gzip.open(gz_path, "rb") as f_in:
            with open(out_path, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)
        os.remove(gz_path)

    print("Done.")


if __name__ == "__main__":
    main()
