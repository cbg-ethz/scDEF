import matplotlib as mpl
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

df = pd.read_csv(snakemake.input[0])
df = df.rename(columns = {'frac_shared':'Fraction of shared cell groups', 'value':'Adjusted rand score', 'method': "Method"})

axx = sns.catplot(data=df, x="Fraction of shared cell groups", y="Adjusted rand score", hue="Method", col="n_batches", kind="box", boxprops=dict(alpha=.3))
colors = []
box_patches = [patch for patch in axx.axes[0][0].patches if type(patch) == mpl.patches.PathPatch]
for i, box in enumerate(box_patches):
    colors.append(box.get_facecolor())

lines_per_boxplot = 6
for ax in axx.axes[0]:
    box_patches = [patch for patch in ax.patches if type(patch) == mpl.patches.PathPatch]
    for i, box in enumerate(box_patches):
        color = box.get_facecolor()
        box.set_color(color)
        for lin in ax.lines[i * lines_per_boxplot: (i + 1) * lines_per_boxplot]:
            lin.set_color(color)
            lin.set_markerfacecolor(color)
            lin.set_markeredgecolor(color)

n_batches = np.unique(df["n_batches"])
axes = axx.axes.flatten()
for i, ax in enumerate(axes):
    print(i)
    ax.set_title(f"{n_batches[i]} batches")

axx2 = axx.map_dataframe(sns.swarmplot,  x="Fraction of shared cell groups", y="Adjusted rand score", hue="Method", dodge=True, palette=colors)
axes = axx2.axes.flatten()
for i, ax in enumerate(axes):
    print(i)
    ax.set_title(f"{n_batches[i]} batches")

axx2.savefig(snakemake.output[0])
