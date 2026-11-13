"""Decompose the kept scDEF model with batch annotations of one four-batch data set (added 2026-11-05).

    python decompose.py <id>          (env: scdef; GPU when visible)

As refit_best_of_seeds.main does for the real data: ``scDEF.decompose_batch_effects(model, top_layer=2)``, then
``scd.tl.filter(dec, batch_key="Batch", **DECOMPOSED)``, from the kept best-of-five fit with batch annotations
(scdef/scDEF/model, reloaded on simcommon.scdef_adata). Writes scdef/scDEF_decomposed/{fit.json, model/}; fit.json
records the decomposition's minutes and the source fit's seed and minutes (runtime = kept seed + decomposition, as the
real-data scorer). Resumable: skips when fit.json says ok.
"""
import os
import sys
import time

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
import simcommon as S  # noqa: E402

sys.path.insert(0, S.MAKE_FIGURES)
import refit_best_of_seeds as rb  # noqa: E402
import scdef as scd  # noqa: E402
from fit_scdef import save  # noqa: E402


def main(ds, top_layer=2):
    import jax

    if not S.parse_id(ds)["multi"]:
        raise SystemExit(f"{ds} has one batch: nothing to decompose")
    src = S.ds_dir(ds, "scdef", "scDEF")
    src_log = S.read_json(os.path.join(src, "fit.json"), {})
    label = S.DECOMPOSED if top_layer == 2 else S.DECOMPOSED_T1
    d = S.ds_dir(ds, "scdef", label)
    log_path = os.path.join(d, "fit.json")
    if S.read_json(log_path, {}).get("status") == "ok" and os.path.isdir(os.path.join(d, "model")):
        print(f"skip {ds}/{label}: ok")
        return 0
    if src_log.get("status") != "ok":
        raise SystemExit(f"{ds}: the fit with batch annotations is not finished")
    log = dict(id=ds, label=label, status="running", source="scdef/scDEF (kept best-of-five, batch key Batch)",
               source_seed=src_log["chosen_seed"], method=f"decompose_batch_effects(top_layer={top_layer}), then "
               "scd.tl.filter(batch_key='Batch', **refit_best_of_seeds.DECOMPOSED)", filter=rb.DECOMPOSED,
               device=str(jax.devices()[0]), cpu_affinity=sorted(os.sched_getaffinity(0)))
    S.write_json(log, log_path)
    a = S.scdef_adata(ds)
    m = scd.scDEF.load(os.path.join(src, "model"), adata=a)
    t0 = time.time()
    dec = scd.scDEF.decompose_batch_effects(m, top_layer=top_layer)
    scd.tl.filter(dec, batch_key="Batch", **rb.DECOMPOSED)
    minutes = (time.time() - t0) / 60
    save(dec, os.path.join(d, "model"))
    log.update(status="ok", decomposition_minutes=round(minutes, 2), layer_sizes=[len(f) for f in dec.factor_lists],
               layer_names=list(dec.layer_names), decomposition_top_layer=dec.decomposition_top_layer,
               chosen_seed=src_log["chosen_seed"],
               kept_seed_minutes=round(src_log["kept_seed_minutes"] + minutes, 2),
               five_seed_minutes=round(src_log["five_seed_minutes"] + minutes, 2),
               runtime_note="kept_seed_minutes and five_seed_minutes include the decomposition",
               finished=time.strftime("%Y-%m-%d %H:%M:%S"))
    S.write_json(log, log_path)
    print(f"done {ds}/{label}: {minutes:.1f} min, layer sizes {log['layer_sizes']}", flush=True)
    return 0


if __name__ == "__main__":
    tl = int(sys.argv[sys.argv.index("--top-layer") + 1]) if "--top-layer" in sys.argv else 2
    sys.exit(main(sys.argv[1], tl))
