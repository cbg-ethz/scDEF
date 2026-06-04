"""Download planaria (S. mediterranea) data from the Plass et al. 2018 web portal."""
import os
import urllib.request

BASE_URL = "https://bimsbstatic.mdc-berlin.de/rajewsky/PSCA"
FILES = {
    "dge.txt": f"{BASE_URL}/dge.txt.gz",
    "R_pca_seurat.txt": f"{BASE_URL}/R_pca_seurat.txt",
    "R_annotation.txt": f"{BASE_URL}/R_annotation.txt",
    "colors_dataset.txt": f"{BASE_URL}/colors_dataset.txt",
}


def main():
    out_dir = snakemake.params["out_dir"]
    os.makedirs(out_dir, exist_ok=True)

    for fname, url in FILES.items():
        out_path = os.path.join(out_dir, fname)
        if url.endswith(".gz"):
            import gzip
            import shutil

            gz_path = out_path + ".gz"
            print(f"Downloading {url} ...")
            urllib.request.urlretrieve(url, gz_path)
            with gzip.open(gz_path, "rb") as f_in:
                with open(out_path, "wb") as f_out:
                    shutil.copyfileobj(f_in, f_out)
            os.remove(gz_path)
        else:
            print(f"Downloading {url} ...")
            urllib.request.urlretrieve(url, out_path)

    with open(snakemake.output["done"], "w") as f:
        f.write("done\n")


if __name__ == "__main__":
    main()
