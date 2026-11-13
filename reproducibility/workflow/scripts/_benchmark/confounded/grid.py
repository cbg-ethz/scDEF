"""The pre-registered grid, and the aggregation of its results.

    python -m confounded.grid list                 # every dataset spec, in run order
    python -m confounded.grid run [--only PREFIX]  # run what has no results.json yet
    python -m confounded.grid summary              # aggregate into summary.csv / verdicts.csv

Runs are sequential (one GPU). A finished dataset is skipped, so the grid can be stopped and
resumed. A failed dataset is logged and the grid moves on.
"""
import argparse
import glob
import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd

from .run import DEFAULT_OUT

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEEDS = [0, 1, 2]
TECHS = ["T1", "T2", "T3"]


def specs():
    out = []
    for s in SEEDS:
        for loc in ["1", "3", "all"]:
            for n in [50]:          # Amendment 1: the 500-gene arm is dropped
                for t in TECHS:
                    out.append(f"D1_loc{loc}_n{n}_{t}_s{s}")
        for t in TECHS:
            out.append(f"D2_{t}_s{s}")
        for p in range(4):
            out.append(f"D2b_p{p}_s{s}")
        out.append(f"D3_s{s}")
    return out


def run(out_root, only=None, extra=()):
    log = os.path.join(out_root, "grid.log")
    os.makedirs(out_root, exist_ok=True)
    for spec in specs():
        if only and not spec.startswith(only):
            continue
        if os.path.exists(os.path.join(out_root, spec, "results.json")):
            continue
        with open(os.path.join(out_root, f"{spec}.log"), "w") as f:
            rc = subprocess.run([sys.executable, "-m", "confounded.run", spec, "--out", out_root, *extra],
                                cwd=HERE, stdout=f, stderr=subprocess.STDOUT).returncode
        with open(log, "a") as f:
            f.write(f"{spec}\t{'ok' if rc == 0 else f'FAILED rc={rc}'}\n")


def _config(meta):
    d = meta["design"]
    if d == "D1":
        return f"D1 loc={meta['localisation']} n={meta['n_resp']} {meta['tech']}"
    if d == "D2":
        return f"D2 {meta['tech']}"
    if d == "D2b":
        return f"D2b {'-'.join(meta['pair'])}"
    return d


def summary(out_root):
    rows = []
    for fn in sorted(glob.glob(os.path.join(out_root, "*", "results.json"))):
        r = json.load(open(fn))
        base = dict(dataset=r["dataset"], config=_config(r["meta"]), seed=r["meta"]["seed"])
        for m, v in r["calibration"].items():
            rows.append(dict(base, method=m, metric="calibration_mae", value=v["mae"]))
            rows.append(dict(base, method=m, metric="calibration_spearman", value=v["spearman"]))
        for m, v in r["attribution"].items():
            for k in ("bio_auprc", "tech_auprc"):
                if k in v:
                    rows.append(dict(base, method=m, metric=k, value=v[k]))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_root, "long.csv"), index=False)
    agg = df.groupby(["config", "metric", "method"])["value"].agg(["mean", "std", "count"])
    agg.to_csv(os.path.join(out_root, "summary.csv"))

    # Pre-registered decision rule: better on the mean and in every seed.
    lower_better = {"calibration_mae"}
    variants = ["scDEF_default", "scDEF_decomposed_raw", "scDEF_decomposed_shape"]
    comparisons = [(v, c) for v in variants for c in ("Harmony", "Harmony_DE", "Harmony_DE_rule")]
    verdicts = []
    for (cfg, metric), g in df.groupby(["config", "metric"]):
        if metric == "calibration_spearman":
            continue
        piv = g.pivot_table(index="seed", columns="method", values="value")
        for a, b in comparisons:
            if a not in piv or b not in piv:
                continue
            diff = piv[a] - piv[b]
            if metric in lower_better:
                diff = -diff
            n = int(diff.notna().sum())
            verdicts.append(dict(
                config=cfg, metric=metric, method=a, versus=b, seeds=n,
                mean_a=piv[a].mean(), mean_b=piv[b].mean(),
                verdict=("outperforms" if n == len(SEEDS) and (diff > 0).all() else
                         "underperforms" if n == len(SEEDS) and (diff < 0).all() else
                         "mixed"),
            ))
    pd.DataFrame(verdicts).to_csv(os.path.join(out_root, "verdicts.csv"), index=False)
    return agg


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["list", "run", "summary"])
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--only", default=None)
    ap.add_argument("--workflow", choices=["relaxed", "learn"], default="relaxed")
    ap.add_argument("--reuse", default=None)
    a = ap.parse_args()
    if a.cmd == "list":
        print("\n".join(specs()))
        print(len(specs()), "datasets", file=sys.stderr)
    elif a.cmd == "run":
        extra = ["--workflow", a.workflow] + (["--reuse", a.reuse] if a.reuse else [])
        run(a.out, a.only, extra)
    else:
        print(summary(a.out).to_string())
