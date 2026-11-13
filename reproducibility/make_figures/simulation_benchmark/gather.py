"""Gather the per-data-set score tables of the simulation benchmark into one table each.

    python gather.py          (any env with pandas)

Writes to out/simulation_benchmark/: scores_all.csv (2024 metrics, naming fix on), scores_all_2024_verbatim.csv,
level_scores_all.csv (level recovery, level signatures), layer_sizes_all.csv, batch_groups_all.csv (shared and
batch-specific groups, four batches), status_all.csv (every data set x
method: finished or not, runtime, device), and summary_medians.csv (median over replicates per scenario, similarity,
method and metric, for reading).
"""
import os

import pandas as pd

import simcommon as S

TABLES = {"scores.csv": "scores_all.csv", "scores_2024_verbatim.csv": "scores_all_2024_verbatim.csv",
          "level_scores.csv": "level_scores_all.csv", "layer_sizes.csv": "layer_sizes_all.csv",
          "batch_groups.csv": "batch_groups_all.csv"}


def main():
    acc = {k: [] for k in TABLES}
    status = []
    for ds in S.all_ids():
        d = os.path.join(S.OUT_ROOT, ds)
        p = S.parse_id(ds)
        for src in TABLES:
            f = os.path.join(d, src)
            if os.path.exists(f) and os.path.getsize(f) > 1:
                try:
                    acc[src].append(pd.read_csv(f))
                except pd.errors.EmptyDataError:
                    pass
        for m in S.scored_methods(ds):
            r = S.read_json(os.path.join(d, "methods", m, "run.json"), {})
            status.append(dict(id=ds, scenario=p["scenario"], de_fscale=p["de_fscale"], rep=p["rep"], method=m,
                               status=r.get("status", "not run"), runtime_s=r.get("runtime_s"),
                               device="gpu" if r.get("cuda_visible_devices") not in (None, "") else "cpu",
                               error=(r.get("error") or "")[:300]))
        for lb in S.scdef_fits(ds) | ({S.DECOMPOSED: "Batch"} if p["multi"] else {}):
            r = S.read_json(os.path.join(d, "scdef", lb, "fit.json"), {})
            status.append(dict(id=ds, scenario=p["scenario"], de_fscale=p["de_fscale"], rep=p["rep"], method=lb,
                               status=r.get("status", "not run"),
                               runtime_s=60 * r["kept_seed_minutes"] if "kept_seed_minutes" in r else None,
                               device=r.get("device"), seeds_done=len(r.get("seeds", []))))
    for src, dst in TABLES.items():
        if acc[src]:
            pd.concat(acc[src], ignore_index=True).to_csv(os.path.join(S.OUT_ROOT, dst), index=False)
    st = pd.DataFrame(status)
    st.to_csv(os.path.join(S.OUT_ROOT, "status_all.csv"), index=False)
    if acc["scores.csv"]:
        sc = pd.concat(acc["scores.csv"] + acc["level_scores.csv"], ignore_index=True)
        sc["metric"] = sc.apply(lambda r: f"{r['metric']} ({r['true_level']})"
                                if isinstance(r.get("true_level"), str) else r["metric"], axis=1)
        sc = sc[~sc["metric"].str.startswith("Learned")]
        med = sc.groupby(["scenario", "similarity", "metric", "method"])["value"].median().unstack("method")
        med.to_csv(os.path.join(S.OUT_ROOT, "summary_medians.csv"))
    print(st.groupby(["status"]).size().to_string())
    print("tables in", S.OUT_ROOT)


if __name__ == "__main__":
    main()
