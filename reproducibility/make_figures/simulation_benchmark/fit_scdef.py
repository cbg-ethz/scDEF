"""Fit scDEF on one simulated data set with the fixed workflow from five seeds and keep the lowest final loss.

    python fit_scdef.py <id> <label>     label: scDEF (with the batch key on four batches) | scDEF_un | scDEF_1L
                                         (env: scdef)

``refit_best_of_seeds.fit_fixed(a, seed, batch_key)`` for seeds 42, 1, 2, 3, 4 on ``simcommon.scdef_adata(id)``;
the fit with the lowest ``final_loss`` (mean of the last 100 iterations of the last fit) is kept, as for every real
data set (decision-fixed-workflow). The choice never looks at the groups. The decomposition is decompose.py.

scDEF_1L (added 2026-11-05, the one-layer baseline of hc.py): the same steps as fit_fixed (fit, FIXED_FIRST filter,
refit, FIXED_SECOND filter; hierarchical_init) and the same five seeds and rule, with ``layer_sizes=[100]`` (one layer
of the default n_factors) and the batch key on four batches.

Writes <id>/scdef/<label>/{fit.json, model/}: every seed's loss, minutes, layer sizes and ARI of each layer with
GroupC (``refit_best_of_seeds.describe``), the device, and the kept model (saved without its AnnData; scoring
rebuilds it with scdef_adata). Resumable: the best fit so far is saved after each seed, and a restart skips the seeds
already logged. Device: JAX's default (GPU when visible; run_all.sh sets JAX_PLATFORMS=cpu when the GPU is taken).
"""
import argparse
import os
import shutil
import sys
import time

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
import simcommon as S  # noqa: E402

sys.path.insert(0, S.MAKE_FIGURES)
import refit_best_of_seeds as rb  # noqa: E402
import scdef as scd  # noqa: E402


def save(m, path):
    # as refit_best_of_seeds.save: clear the fused-step caches (unpicklable closures) before saving
    for attr in ("_learn_step_cache", "_learn_jit_cache"):
        if hasattr(m, attr):
            setattr(m, attr, None)
    tmp = path + ".tmp"
    if os.path.isdir(tmp):
        shutil.rmtree(tmp)
    m.save(tmp, overwrite=True)
    if os.path.isdir(path):
        shutil.rmtree(path)
    os.replace(tmp, path)


def fit_one_layer(a, seed, batch_key=None):
    """refit_best_of_seeds.fit_fixed with one layer: identical steps and settings, layer_sizes=[100]."""
    m = scd.scDEF(a.copy(), counts_layer="counts", batch_key=batch_key, seed=seed, layer_sizes=S.ONE_LAYER_SIZES)
    m.fit(hierarchical_init=True)
    scd.tl.filter(m, batch_key=batch_key, **rb.FIXED_FIRST)
    m.fit(hierarchical_init=True)
    scd.tl.filter(m, batch_key=batch_key, **rb.FIXED_SECOND)
    return m


def main(ds, label):
    import jax

    variants = S.scdef_fits(ds)
    if label not in variants:
        raise SystemExit(f"{label} is not fitted on {ds}; one of {list(variants)}")
    batch_key = variants[label]
    d = S.ds_dir(ds, "scdef", label)
    log_path = os.path.join(d, "fit.json")
    model_path = os.path.join(d, "model")
    log = S.read_json(log_path, {})
    if log.get("status") == "ok" and os.path.isdir(model_path):
        print(f"skip {ds}/{label}: ok")
        return 0
    if not os.path.isdir(model_path):          # nothing kept yet: start over
        log = {}
    if not log:
        log = dict(id=ds, label=label, batch_key=batch_key, seeds_planned=S.SCDEF_SEEDS, rule="lowest final loss",
                   workflow="refit_best_of_seeds.fit_fixed (FIXED_FIRST, refit, FIXED_SECOND)" if label != S.ONE_LAYER
                   else "fit_fixed's steps with layer_sizes=%s (fit_scdef.fit_one_layer)" % S.ONE_LAYER_SIZES,
                   first_filter=rb.FIXED_FIRST, second_filter=rb.FIXED_SECOND, seeds=[])
    log.update(status="running", device=str(jax.devices()[0]), jax_platforms=os.environ.get("JAX_PLATFORMS"),
               cpu_affinity=sorted(os.sched_getaffinity(0)), scdef_version=scd.__version__)
    S.write_json(log, log_path)
    a = S.scdef_adata(ds)
    done = {r["seed"] for r in log["seeds"]}
    best = log.get("best_so_far")
    for seed in S.SCDEF_SEEDS:
        if seed in done:
            continue
        t0 = time.time()
        if label == S.ONE_LAYER:
            m = fit_one_layer(a, seed, batch_key=batch_key)
        else:
            m = rb.fit_fixed(a, seed, batch_key=batch_key)
        rec = rb.describe(m, seed, (time.time() - t0) / 60)
        rec["layer_names"] = list(m.layer_names)
        rec["device"] = str(jax.devices()[0])
        log["seeds"].append(rec)
        print(label, rec, flush=True)
        if best is None or rec["final_loss"] < best["final_loss"]:
            save(m, model_path)
            best = dict(seed=seed, final_loss=rec["final_loss"])
            log["best_so_far"] = best
        S.write_json(log, log_path)
        del m
    log.update(status="ok", chosen_seed=best["seed"], finished=time.strftime("%Y-%m-%d %H:%M:%S"),
               kept_seed_minutes=[r["minutes"] for r in log["seeds"] if r["seed"] == best["seed"]][0],
               five_seed_minutes=round(sum(r["minutes"] for r in log["seeds"]), 2))
    S.write_json(log, log_path)
    print(f"done {ds}/{label}: seed {best['seed']} kept", flush=True)
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("id")
    ap.add_argument("label")
    a = ap.parse_args()
    sys.exit(main(a.id, a.label))
