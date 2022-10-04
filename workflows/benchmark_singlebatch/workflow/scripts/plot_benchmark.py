import matplotlib as mpl
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv(snakemake.input[0])

ax = sns.boxplot(data=df, x="de_fscale", y="value", hue="method", boxprops=dict(alpha=.3))
# change boxplot colors
lines_per_boxplot = 6
box_patches = [patch for patch in ax.patches if type(patch) == mpl.patches.PathPatch]
for i, box in enumerate(box_patches):
    color = box.get_facecolor()
    box.set_color(color)
    for lin in ax.lines[i * lines_per_boxplot: (i + 1) * lines_per_boxplot]:
        lin.set_color(color)
        lin.set_markerfacecolor(color)
        lin.set_markeredgecolor(color)
sns.swarmplot(data=df, x="de_fscale", y="value", hue="method", dodge=True, ax=ax, alpha=0.8)
# remove extra legend handles and rename
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles[:3], labels[:3], title='Method',loc='upper left')
ax.set(xlabel='Simulated DE scale', ylabel='Adjusted rand score')

sns.despine()

plt.savefig(snakemake.output[0])
