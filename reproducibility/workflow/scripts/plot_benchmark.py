import matplotlib as mpl
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

df1 = pd.read_csv(snakemake.input[0])
df2 = pd.read_csv(snakemake.input[1])

# Merge batches
df1["name"] = "Single batch"


def translate(l):
    if "1" in l[0]:
        return "Four similar batches"
    else:
        return "Four distinct batches"


df2 = df2.assign(name=df2[["frac_shared"]].apply(lambda row: translate(row), axis=1))
df = pd.concat([df1, df2], axis=0)

df = df.replace({"Unintegrated": "Leiden"})

annots = {
    "scDEF": "True",
    "scDEF_un": "False",
    "NMF": "False",
    "scHPF": "False",
    "Unintegrated": "False",
    "scVI": "True",
    "Harmony": "True",
    "Scanorama": "True",
}

colordict = {
    "scDEF": "green",
    "scDEF (-)": "lightgreen",
    "NMF": "blue",
    "scHPF": "darkblue",
    "Leiden": "violet",
    "scVI": "orange",
    "Harmony": "red",
    "Scanorama": "darkred",
}

for metric in [
    "Cell Type ARI",
    "Signature accuracy",
    "Hierarchy accuracy",
    "Hierarchical signature consistency",
]:
    ax = sns.catplot(
        data=df,
        kind="box",
        x="Separability",
        y=metric,
        hue="Method",
        col="name",
        style="Annotations",
        palette=colordict,
        hue_order=[
            "scDEF",
            "scDEF_un",
            "scVI",
            "Harmony",
            "Scanorama",
            "Leiden",
            "scHPF",
            "NMF",
        ],
        aspect=0.5,
        dodge=True,
    )
    ax.set_titles(name)
    ax.fig.set_size_inches(12, 4)
    plt.savefig(f"{metric_fname[metric]}.pdf")
