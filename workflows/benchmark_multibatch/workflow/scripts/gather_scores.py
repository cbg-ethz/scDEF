import sys
import os
import glob

collectionName = snakemake.input["fname_list"][0]

for _ in range(4):  # go up to results/
    collectionName = os.path.dirname(collectionName)

outFileName = snakemake.output[0]

pattern = (
    collectionName + "/**/**/**/*score.txt"
)  # all files matching this pattern are processed

fileList = glob.glob(pattern, recursive=False)

rows = []
for filename in fileList:
    # Parse filename
    l = filename.split("/")
    method = l[1]
    n_batches = l[2]
    frac_shared = l[3]
    rep_id, score, _ = l[4].split("_")

    # Parse scores
    with open(filename) as f:
        lines = f.read().splitlines()
        value = lines[0]

    rows.append(
        [
            method,
            n_batches,
            frac_shared,
            rep_id,
            score,
            value,
        ]
    )

columns = [
    "method",
    "n_batches",
    "frac_shared",
    "rep_id",
    "score",
    "value",
]

import pandas as pd

scores = pd.DataFrame.from_records(rows, columns=columns)
scores = pd.melt(
    scores,
    id_vars=[
        "method",
        "n_batches",
        "frac_shared",
        "rep_id",
        "score",
    ],
    value_vars="value",
)
scores.to_csv(outFileName, index=False)
