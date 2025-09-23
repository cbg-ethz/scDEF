import pandas as pd
import numpy as np

fileList = snakemake.input["fname_list"]
rows = []
for filename in fileList:
    # Parse filename
    # /{method}/sep_{separability}/shared_{frac_shared}/rep_{rep_id}_scores.csv',
    l = filename.split("/")
    method = l[-4]
    separability = l[-3].split("_")[1]
    frac_shared = l[-2].split("_")[1]
    rep_id = l[-1].split("_")[1]

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

scores.to_csv(snakemake.output["fname"], index=False)
