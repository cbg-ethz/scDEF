import pandas as pd
import numpy as np


def main():
    rows = []
    param_names = snakemake.params["param_name"]
    for filename in snakemake.input["fname_list"]:
        print(filename)

        # Parse filename
        l = filename.split("/")

        rep_id = l[-1].split("_")[1]
        method = l[snakemake.params["method_idx"]]
        if param_names != "":
            params = []
            for param_idx in snakemake.params["param_idx"]:
                params.append(l[param_idx].split("_")[1])

        # Parse scores
        df = pd.read_csv(filename, index_col=0)
        print(df)

        for idx, score in enumerate(df.index):  # must have the ARI per layer
            value = df.values[idx]
            value = value[0]
            if isinstance(value, str):
                value = np.mean(np.array(value.strip("][").split(", ")).astype(float))
            value = float(value)
            if param_names != "":
                rows.append(
                    [
                        method,
                        *params,
                        rep_id,
                        score,
                        value,
                    ]
                )
            else:
                rows.append(
                    [
                        method,
                        rep_id,
                        score,
                        value,
                    ]
                )

    if param_names != "":
        columns = [
            "method",
            *param_names,
            "rep_id",
            "score",
            "value",
        ]
    else:
        columns = [
            "method",
            "rep_id",
            "score",
            "value",
        ]

    scores = pd.DataFrame.from_records(rows, columns=columns)
    print(scores)

    scores.to_csv(snakemake.output[0], index=False)


if __name__ == "__main__":
    main()
