import sys
import os
import glob

collectionName = snakemake.input["fname_list"][0]

for _ in range(3):  # go up to results/
    collectionName = os.path.dirname(collectionName)

outFileName = snakemake.output[0]

pattern = (
    collectionName + "/**/**/*mod.txt"
)  # all files matching this pattern are processed

fileList = glob.glob(pattern, recursive=False)

rows = []
for filename in fileList:
    # Parse filename
    l = filename.split("/")
    method = l[1]
    de_fscale = l[2]
    rep_id = l[3].split("_")[0]

    # Parse scores
    with open(filename) as f:
        lines = f.read().splitlines()
        ari = lines[0]

    rows.append(
        [
            method,
            de_fscale,
            rep_id,
            ari,
        ]
    )

columns = [
    "method",
    "de_prob",
    "rep_id",
    "mod",
]

import pandas as pd

scores = pd.DataFrame.from_records(rows, columns=columns)
scores = pd.melt(
    scores,
    id_vars=[
        "method",
        "de_prob",
        "rep_id",
    ],
    value_vars="mod",
    var_name="score",
)
scores.to_csv(outFileName, index=False)
