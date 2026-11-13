#!/usr/bin/env bash
# Local re-run of the real-data benchmarks: every comparator re-run on this machine, scDEF scored from the
# saved best-of-five models. Resumable: finished steps are skipped (delete a method's run.json to redo it).
#
#   bash run_all.sh                       # pbmcs2b pbmcsifn planaria (the 3k PBMCs were the end-to-end test)
#   bash run_all.sh pbmcs3k planaria      # any subset, in order
#
# Per data set: prepare (env scdef) -> each method (env scdef-bench2, one process each, sequential so that
# runtimes are comparable, capped at BENCH_BUDGET_HOURS, default 4 as in the paper) -> score (env scdef)
# -> plot. GPU (scVI, f-scLVM/MuVI only) is used only if no other process holds it when the method starts;
# otherwise the method runs on CPU and run.json records that. CPU work is pinned to cores 8-31 at nice 10.
# Planaria signature accuracy needs the 2024 marker file: PLANARIA_MARKERS=/path/planaria_marker_genes.csv
# (default ~/projects/scDEF/scdef_data/planaria/planaria_marker_genes.csv); without it that metric is empty.
#
# Logs: out/realdata_benchmark/<data>/methods/<method>/log.txt and out/realdata_benchmark/<data>/run_all.log
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"
SCDEF_PY="${SCDEF_PY:-$HOME/miniconda3/envs/scdef/bin/python}"
BENCH_PY="${BENCH_PY:-$HOME/miniconda3/envs/scdef-bench2/bin/python}"
BUDGET_HOURS="${BENCH_BUDGET_HOURS:-4}"
export BENCH_BUDGET_HOURS="$BUDGET_HOURS"
CPUS="${BENCH_CPUS:-8-31}"
PIN=(taskset -c "$CPUS" nice -n 10)
OUT="${BENCH_OUT_ROOT:-$HERE/../out/realdata_benchmark}"
DATASETS=("$@")
[ ${#DATASETS[@]} -eq 0 ] && DATASETS=(pbmcs2b pbmcsifn planaria)

gpu_free() {
  command -v nvidia-smi >/dev/null || return 1
  [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d '[:space:]')" ]
}

log() { echo "[$(date '+%F %T')] $*" | tee -a "$OUT/$DATA/run_all.log"; }

for DATA in "${DATASETS[@]}"; do
  DATA="$("$BENCH_PY" -c "import common as C; print(C.canonical('$DATA'))")" || exit 1
  mkdir -p "$OUT/$DATA"
  log "== $DATA: prepare"
  JAX_PLATFORMS=cpu "${PIN[@]}" "$SCDEF_PY" prepare.py "$DATA" >>"$OUT/$DATA/run_all.log" 2>&1 \
    || { log "prepare failed"; continue; }
  METHODS="$("$BENCH_PY" -c "import common as C; print(' '.join(C.DATASETS['$DATA']['methods']))")"
  GPU_METHODS="$("$BENCH_PY" -c "import common as C; print(' '.join(sorted(C.GPU_METHODS)))")"
  for M in $METHODS; do
    D="$OUT/$DATA/methods/$M"
    mkdir -p "$D"
    ST="$("$BENCH_PY" -c "import json,os; p='$D/run.json'; print(json.load(open(p)).get('status','') if os.path.exists(p) else '')")"
    if [ "$ST" = "ok" ] || [ "$ST" = "timeout" ]; then log "skip $M ($ST)"; continue; fi
    DEV=""
    if [[ " $GPU_METHODS " == *" $M "* ]]; then
      nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv >>"$D/log.txt" 2>&1
      if gpu_free; then DEV=0; else log "GPU busy: $M on CPU"; fi
    fi
    log "run $M (device: ${DEV:-cpu}, budget ${BUDGET_HOURS} h)"
    CUDA_VISIBLE_DEVICES="$DEV" timeout --kill-after=120 "${BUDGET_HOURS}h" \
      "${PIN[@]}" "$BENCH_PY" run_method.py "$DATA" "$M" >>"$D/log.txt" 2>&1
    RC=$?
    if [ $RC -eq 124 ] || [ $RC -eq 137 ]; then
      "$BENCH_PY" run_method.py "$DATA" "$M" --mark-timeout >>"$D/log.txt" 2>&1
      log "$M exceeded ${BUDGET_HOURS} h"
    elif [ $RC -ne 0 ]; then
      log "$M failed (exit $RC); see $D/log.txt"
    else
      log "$M done: $(tail -1 "$D/log.txt")"
    fi
  done
  log "score"
  "${PIN[@]}" "$SCDEF_PY" score.py "$DATA" >>"$OUT/$DATA/run_all.log" 2>&1 || log "score failed"
  log "plot"
  "$SCDEF_PY" plot.py "$DATA" >>"$OUT/$DATA/run_all.log" 2>&1 || log "plot failed"
  log "== $DATA finished"
done
