import pandas as pd
import numpy as np


def main():
    rows = []
    for filename in snakemake.input["fname_list"]:
        print(filename)

        l = filename.split("/")
        method = l[-2]
        # Filename is seed_{seed}_scores.csv
        seed = l[-1].split("_")[1]

        df = pd.read_csv(filename, index_col=0)
        print(df)

        for idx, score in enumerate(df.index):
            value = df.values[idx]
            value = value[0]
            if isinstance(value, str):
                value = np.mean(np.array(value.strip("][").split(", ")).astype(float))
            value = float(value)
            rows.append([method, seed, score, value])

    scores = pd.DataFrame.from_records(
        rows, columns=["method", "seed", "score", "value"]
    )
    print(scores)
    scores.to_csv(snakemake.output["fname"], index=False)


if __name__ == "__main__":
    main()
