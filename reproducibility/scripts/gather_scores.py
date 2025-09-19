import sys
import os
import glob
import pandas as pd
import numpy as np

collectionName = snakemake.input["fname_list"][0]

for _ in range(1):  # go up to results/
    collectionName = os.path.dirname(collectionName)

outFileName = snakemake.output[0]

pattern = collectionName + "/**/*.csv"  # all files matching this pattern are processed

fileList = glob.glob(pattern, recursive=False)

rows = []
for filename in fileList:
    # Parse filename
    l = filename.split("/")
    method = l[1]
    separability = l[2].split("_")[1]
    frac_shared = l[3].split("_")[1]
    rep_id = l[4].split("_")[1]

    print(filename)

    # Parse scores
    df = pd.read_csv(filename, index_col=0)
    print(df)
    for idx, score in enumerate(df.index):
        value = df.values[idx]
        value = value[0]
        if isinstance(value, str):
            value = np.mean(np.array(value.strip("][").split(", ")).astype(float))
        value = float(value)
        rows.append(
            [
                method,
                separability,
                frac_shared,
                rep_id,
                score,
                value,
            ]
        )

columns = [
    "method",
    "separability",
    "frac_shared",
    "rep_id",
    "score",
    "value",
]

scores = pd.DataFrame.from_records(rows, columns=columns)
print(scores)

scores = pd.melt(
    scores,
    id_vars=[
        "method",
        "separability",
        "frac_shared",
        "rep_id",
        "score",
    ],
    value_vars="value",
)
scores.to_csv(outFileName, index=False)
