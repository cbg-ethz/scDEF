"""Simulate one data set with simulate.R (the 2024 generator plus additions) in the scdef-splatter environment.

    python simulate.py <id> [--umap] [--force]        e.g. single_fs2.0_r0, similar_fs0.5_r3, distinct_fs2.0_r1

Parameters as the 2024 rules generate_{single,multi}batch_data. Writes <id>/data/{counts,meta,markers,defacs,
groups}.csv (and the UMAP PNGs with --umap) and simulate.json. Skips if simulate.json records success and the files
exist. A data set simulated before groups.csv existed (2026-11-05, first launch) is regenerated into data/regen/; its
groups.csv is kept only if the regenerated counts, meta, markers and DE factors are byte-identical to the ones in use
(otherwise exit code 4 and nothing changes).
"""
import argparse
import filecmp
import os
import shutil
import subprocess
import sys
import time

import simcommon as S

RSCRIPT = os.environ.get("SPLATTER_RSCRIPT", os.path.expanduser("~/miniconda3/envs/scdef-splatter/bin/Rscript"))
DATA_FILES = ("counts", "meta", "markers", "defacs")


def run_r(p, d, umap):
    out = {k: os.path.join(d, f"{k}.csv") for k in DATA_FILES + ("groups",)}
    args = dict(n_cells=p["n_cells"], n_batches=p["n_batches"], frac_shared=p["frac_shared"],
                de_fscale=p["de_fscale"], de_prob=p["de_prob"], batch_facscale=p["batch_facscale"], coverage=0,
                seed=p["rep"], counts_fname=out["counts"], meta_fname=out["meta"], markers_fname=out["markers"],
                defacs_fname=out["defacs"], groups_fname=out["groups"], reference=S.REF_MTX,
                params_rds=S.PARAMS_RDS, umap="TRUE" if umap else "FALSE", umap_fname=os.path.join(d, "umap.png"),
                umap_nobatch_fname=os.path.join(d, "umap_nobatch.png"))
    cmd = [RSCRIPT, os.path.join(S.HERE, "simulate.R")] + [f"{k}={v}" for k, v in args.items()]
    t0 = time.time()
    rc = subprocess.call(cmd, cwd=d)
    return rc, args, out, round(time.time() - t0, 1)


def main(ds, umap=False, force=False):
    p = S.parse_id(ds)
    d = S.ds_dir(ds, "data")
    out = {k: os.path.join(d, f"{k}.csv") for k in DATA_FILES + ("groups",)}
    log_path = os.path.join(d, "simulate.json")
    prev = S.read_json(log_path, {})
    done = prev.get("returncode") == 0 and all(os.path.exists(out[k]) for k in DATA_FILES)
    if not force and done and os.path.exists(out["groups"]):
        print("exists", d)
        return 0
    if not force and done:                       # simulated before groups.csv: regenerate and compare
        rd = os.path.join(d, "regen")
        if os.path.isdir(rd):
            shutil.rmtree(rd)
        os.makedirs(rd)
        rc, _, rout, secs = run_r(p, rd, False)
        same = rc == 0 and all(filecmp.cmp(rout[k], out[k], shallow=False) for k in DATA_FILES)
        prev.update(groups_regenerated=dict(returncode=rc, identical=same, seconds=secs,
                                            when=time.strftime("%Y-%m-%d %H:%M:%S")))
        if same:
            shutil.move(rout["groups"], out["groups"])
            shutil.rmtree(rd)
            S.write_json(prev, log_path)
            print(f"{ds}: regenerated identically; groups.csv kept")
            return 0
        S.write_json(prev, log_path)
        print(f"{ds}: REGENERATED DATA DIFFER (rc {rc}); groups.csv not written, see {rd}", file=sys.stderr)
        return 4
    if os.path.exists(log_path):
        os.remove(log_path)
    for f in out.values():                       # a partial earlier run must not look finished
        if os.path.exists(f):
            os.remove(f)
    rc, args, _, secs = run_r(p, d, umap)
    S.write_json(dict(id=ds, params={k: v for k, v in args.items() if not k.endswith("fname")},
                      returncode=rc, seconds=secs, generator="simulation_benchmark/simulate.R", rscript=RSCRIPT),
                 log_path)
    if rc != 0:
        for f in out.values():
            if os.path.exists(f):
                os.remove(f)
    return rc


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("id")
    ap.add_argument("--umap", action="store_true")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    sys.exit(main(a.id, a.umap, a.force))
