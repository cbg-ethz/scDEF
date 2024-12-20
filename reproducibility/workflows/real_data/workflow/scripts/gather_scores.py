import pandas as pd
import numpy as np

fileList = snakemake.input["fname_list"]
rows = []
for filename in fileList:
    # Parse filename
    l = filename.split("/")
    method = l[-1].split(".csv")[0]

    # Parse scores
    df = pd.read_csv(filename, index_col=0)
    for idx, score in enumerate(df.index):
        value = df.values[idx]
        value = value[0]
        if isinstance(value, str):
            value = np.mean(np.array(value.strip("][").split(", ")).astype(float))
        value = float(value)
        rows.append(
            [
                method,
                score,
                value,
            ]
        )

columns = [
    "method",
    "score",
    "value",
]

scores = pd.DataFrame.from_records(rows, columns=columns)
print(scores)

scores.to_csv(snakemake.output["fname"], index=False)
