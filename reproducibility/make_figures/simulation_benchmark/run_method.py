"""Run one comparator on one simulated data set with the real-data re-run's machinery (no scoring here).

    python run_method.py <id> <method> [--force]          (env: scdef-bench2)
    python run_method.py <id> <method> --mark-timeout     (record that the time budget ran out)

Gene selection, normalisation, the method calls and the saved outputs are those of
``realdata_benchmark/run_method.py`` (``select_genes``, ``run``, ``save``), with two changes for the simulations:

- seed = replicate (the 2024 simulation rules passed ``seed: {rep_id}``), in place of 42;
- f-scLVM (MuVI) reads log-normalised counts, the X of the 2024 simulation input (``prepare_input.py``); the
  real-data inputs of 2024 were regressed and scaled, the simulated ones were not.

With one batch no batch key is given (see simcommon.batch_key). Writes <id>/methods/<method>/{outputs.json,
scores.npz, latent0.npy, run.json}. Resumable: skips a method whose run.json says ``ok`` or ``timeout``.
"""
import argparse
import os
import platform
import sys
import time
import traceback
import warnings

warnings.filterwarnings("ignore")
import numpy as np
import scanpy as sc

import simcommon as S

rd = S.rd_run_method()


def lognorm_only(ad):
    rd.lognorm(ad)
    return "none: log-normalised only (the X of the 2024 simulation input)"


rd.scaled = lognorm_only            # f-scLVM's input (rd.run calls scaled(ad) for fscLVM only)


def load_input(ds):
    a = sc.read_h5ad(os.path.join(S.ds_dir(ds), "input.h5ad"))
    counts = a.layers["counts"]
    counts = counts.toarray() if hasattr(counts, "toarray") else np.asarray(counts)
    a.layers["counts"] = counts.astype(np.float64)       # as the real-data run (2024 inputs held dense counts)
    a.X = a.layers["counts"].copy()
    return a


def main(ds, method, force=False, mark_timeout=False):
    p = S.parse_id(ds)
    if method not in S.methods(ds):
        raise SystemExit(f"{method} is not run on {ds}")
    d = S.ds_dir(ds, "methods", method)
    status_path = os.path.join(d, "run.json")
    prev = S.read_json(status_path, {})
    if mark_timeout:
        S.write_json(dict(prev, method=method, status="timeout", budget_hours=S.rc.BUDGET_HOURS,
                          finished=time.strftime("%Y-%m-%d %H:%M:%S")), status_path)
        return 0
    if prev.get("status") in ("ok", "timeout") and not force:
        print(f"skip {ds}/{method}: {prev['status']}")
        return 0
    rd.C.SEED = p["rep"]                # rd.run reads C.SEED for every seeded method
    info = dict(method=method, data=ds, settings=S.settings(method, p["rep"]), seed=p["rep"], status="running",
                started=time.strftime("%Y-%m-%d %H:%M:%S"), host=platform.node(),
                cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
                cpu_affinity=sorted(os.sched_getaffinity(0)), versions=rd.versions())
    S.write_json(info, status_path)
    try:
        t0 = time.time()
        batch_key = S.batch_key(ds)
        a = load_input(ds)
        a, how = rd.select_genes(a, S.rc.SETTINGS[method]["n_top_genes"], batch_key)
        info.update(genes=how, n_cells=int(a.n_obs), n_genes=int(a.n_vars), batch_key=batch_key)
        outs, runtime = rd.run(method, a, batch_key, info)
        info["preprocess_s"] = round(time.time() - t0 - runtime, 2)
        rd.save(outs, outs.get("adata", a), d, info)
        info.update(status="ok", runtime_s=round(runtime, 2), finished=time.strftime("%Y-%m-%d %H:%M:%S"))
        S.write_json(info, status_path)
        print(f"done {ds}/{method}: {runtime:.1f} s, levels {info['levels']}")
        return 0
    except Exception as e:
        info.update(status="failed", error=repr(e), traceback=traceback.format_exc(),
                    finished=time.strftime("%Y-%m-%d %H:%M:%S"))
        S.write_json(info, status_path)
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("id")
    ap.add_argument("method")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--mark-timeout", action="store_true")
    a = ap.parse_args()
    sys.exit(main(a.id, a.method, a.force, a.mark_timeout))
