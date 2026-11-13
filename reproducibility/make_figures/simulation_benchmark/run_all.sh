#!/usr/bin/env bash
# Simulation benchmark re-run (scDEF-markdown/experiments/exp-simulation-benchmark-rerun.md): simulate -> prepare
# -> scDEF fits (GPU queue) and comparators (CPU queue) in parallel -> score -> gather -> plot. Resumable: every
# step skips what is finished (delete a method's run.json, a fit.json, or a data set's input.h5ad to redo it).
#
#   bash run_all.sh                 # all 30 data sets
#   bash run_all.sh single_fs2.0_r0 # any subset of ids (<scenario>_fs<de_fscale>_r<rep>)
#
# Queues:
#   GPU queue, per data set in order: scDEF best-of-five fit(s) (fit_scdef.py scDEF [scDEF_un]), the decomposition
#     (decompose.py, four batches), the one-layer scDEF (fit_scdef.py scDEF_1L; only with SIMBENCH_ONE_LAYER=1, off
#     since the fixed workflow failed on one layer, 2026-11-05). SIMBENCH_SCDEF_JOBS at a time,
#     default 1: three concurrent fits gave no more epochs per second in total than one (GPU-bound; measured
#     2026-11-05) and would triple each fit's runtime. Pinned to SIMBENCH_SCDEF_CPUS (default 8-13). Each job checks
#     nvidia-smi when it starts: if a process that is not part of this benchmark holds the GPU, that job runs on CPU
#     (JAX_PLATFORMS=cpu; recorded in fit.json).
#   CPU queue: one comparator per data set and method (run_method.py), SIMBENCH_CPU_LANES at a time (default 3),
#     each in its own lane of equal cores out of SIMBENCH_LANE_CPUS (default 14-31: three lanes of six cores), with
#     the thread counts set to the lane size, so every comparator gets the same cores; nice 10; capped at
#     BENCH_BUDGET_HOURS (default 4). scVI and f-scLVM use the GPU unless a foreign process holds it.
#   Post stage, alongside the queues: every few minutes, the hierarchical-clustering baselines (hc.py) whose inputs
#     are ready run in the same lanes; data sets with new results are re-scored (score.py, cached per method, on the
#     scDEF cores), then gather.py and plot.py. Ends after both queues, when no baseline is left to run.
# Logs: <out>/run_all.log (driver); <id>/data/simulate.log, <id>/prepare.log, <id>/methods/<m>/log.txt,
# <id>/scdef/<label>/log.txt, <id>/methods/<hc variant>/log.txt, <id>/score.log, gather.log, plot.log.
# Output: out/simulation_benchmark/.
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"
export HERE
export SCDEF_PY="${SCDEF_PY:-$HOME/miniconda3/envs/scdef/bin/python}"
export BENCH_PY="${BENCH_PY:-$HOME/miniconda3/envs/scdef-bench2/bin/python}"
export BENCH_BUDGET_HOURS="${BENCH_BUDGET_HOURS:-4}"
export OUT="${SIMBENCH_OUT_ROOT:-$HERE/../out/simulation_benchmark}"
export SIMBENCH_OUT_ROOT="$OUT"
N_SCDEF="${SIMBENCH_SCDEF_JOBS:-1}"
export SCDEF_CPUS="${SIMBENCH_SCDEF_CPUS:-8-13}"
export N_LANES="${SIMBENCH_CPU_LANES:-3}"
LANE_CPUS="${SIMBENCH_LANE_CPUS:-14-31}"
N_SIM="${SIMBENCH_SIM_JOBS:-8}"
export ALL_CPUS="${SIMBENCH_ALL_CPUS:-8-31}"
export TQDM_MININTERVAL=30          # fewer progress-bar lines in the logs
mkdir -p "$OUT"
export LOG="$OUT/run_all.log"

log() { echo "[$(date '+%F %T')] $*" >>"$LOG"; echo "[$(date '+%F %T')] $*"; }
export -f log

# lanes: split LANE_CPUS into N_LANES equal core ranges
LO="${LANE_CPUS%-*}"; HI="${LANE_CPUS#*-}"
PER=$(( (HI - LO + 1) / N_LANES ))
export LANE_SIZE="$PER"
LANES=""
for i in $(seq 0 $((N_LANES - 1))); do LANES="$LANES $((LO + i * PER))-$((LO + (i + 1) * PER - 1))"; done
export LANES

# PIDs holding the GPU that are not this benchmark's processes (theirs run with cwd = this directory)
foreign_gpu() {
  command -v nvidia-smi >/dev/null || { echo "no-nvidia-smi"; return; }
  local p
  for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d ' '); do
    [ "$(readlink "/proc/$p/cwd" 2>/dev/null)" = "$HERE" ] || echo "$p"
  done
}
export -f foreign_gpu

take_lane() {     # sets LANE; the flock is held by this shell until it exits
  local fd
  while :; do
    for LANE in $LANES; do
      exec {fd}>"$OUT/.lane_${LANE}.lock"
      if flock -n "$fd"; then return 0; fi
      exec {fd}>&-
    done
    sleep 3
  done
}
export -f take_lane

job_method() {   # <id> <method>
  local ds="$1" m="$2" d="$OUT/$1/methods/$2" dev="" lane
  mkdir -p "$d"
  if grep -qE '"status": "(ok|timeout)"' "$d/run.json" 2>/dev/null; then return 0; fi
  take_lane; lane="$LANE"
  if [ "$m" = scVI ] || [ "$m" = fscLVM ]; then
    if [ -z "$(foreign_gpu)" ]; then dev=0; else log "GPU held by another process: $ds/$m on CPU"; fi
  fi
  log "start $ds/$m (cores $lane, device ${dev:-cpu})"
  CUDA_VISIBLE_DEVICES="$dev" OMP_NUM_THREADS=$LANE_SIZE MKL_NUM_THREADS=$LANE_SIZE \
    OPENBLAS_NUM_THREADS=$LANE_SIZE NUMBA_NUM_THREADS=$LANE_SIZE \
    timeout --kill-after=120 "${BENCH_BUDGET_HOURS}h" taskset -c "$lane" nice -n 10 \
    "$BENCH_PY" run_method.py "$ds" "$m" >>"$d/log.txt" 2>&1
  local rc=$?
  if [ $rc -eq 124 ] || [ $rc -eq 137 ]; then
    "$BENCH_PY" run_method.py "$ds" "$m" --mark-timeout >>"$d/log.txt" 2>&1
    log "$ds/$m exceeded ${BENCH_BUDGET_HOURS} h"
  elif [ $rc -ne 0 ]; then
    log "FAILED $ds/$m (exit $rc); see $d/log.txt"
  else
    log "done $ds/$m: $(tail -1 "$d/log.txt")"
  fi
  return 0
}
export -f job_method

job_scdef() {   # <id> <label>   (label scDEF_decomposed runs decompose.py)
  local ds="$1" lb="$2" d="$OUT/$1/scdef/$2" plat="" script=fit_scdef.py
  mkdir -p "$d"
  if grep -q '"status": "ok"' "$d/fit.json" 2>/dev/null && [ -d "$d/model" ]; then return 0; fi
  [ "$lb" = scDEF_decomposed ] && script=decompose.py
  local foreign; foreign="$(foreign_gpu | tr '\n' ' ')"
  if [ -n "${foreign// /}" ]; then plat=cpu; log "GPU held by $foreign: $ds/$lb on CPU"; fi
  log "start $ds/$lb (cores $SCDEF_CPUS, ${plat:-gpu})"
  if [ "$script" = decompose.py ]; then set -- "$ds"; else set -- "$ds" "$lb"; fi
  env ${plat:+JAX_PLATFORMS=cpu} XLA_PYTHON_CLIENT_PREALLOCATE=false taskset -c "$SCDEF_CPUS" nice -n 10 \
    "$SCDEF_PY" "$script" "$@" >>"$d/log.txt" 2>&1
  local rc=$?
  if [ $rc -ne 0 ]; then log "FAILED $ds/$lb (exit $rc); see $d/log.txt"; else log "done $ds/$lb: $(tail -1 "$d/log.txt")"; fi
  return 0
}
export -f job_scdef

job_simulate() {
  mkdir -p "$OUT/$1/data"
  taskset -c "$ALL_CPUS" nice -n 10 "$SCDEF_PY" simulate.py "$1" >>"$OUT/$1/data/simulate.log" 2>&1 \
    && log "simulated $1" || log "FAILED simulate $1; see $OUT/$1/data/simulate.log"
}
export -f job_simulate

job_prepare() {
  taskset -c "$ALL_CPUS" nice -n 10 "$SCDEF_PY" prepare.py "$1" >>"$OUT/$1/prepare.log" 2>&1
  local rc=$?
  [ $rc -eq 0 ] && log "prepared $1: $(grep nesting "$OUT/$1/prepare.log" | tail -1)" || log "FAILED prepare $1 (exit $rc)"
  return $rc
}
export -f job_prepare

job_hc() {   # <id> <variant>: a hierarchical-clustering baseline, in a lane like the comparators
  local ds="$1" v="$2" d="$OUT/$1/methods/$2" lane
  mkdir -p "$d"
  if grep -q '"status": "ok"' "$d/run.json" 2>/dev/null; then return 0; fi
  take_lane; lane="$LANE"
  log "start $ds/$v (cores $lane)"
  JAX_PLATFORMS=cpu OMP_NUM_THREADS=$LANE_SIZE MKL_NUM_THREADS=$LANE_SIZE OPENBLAS_NUM_THREADS=$LANE_SIZE \
    NUMBA_NUM_THREADS=$LANE_SIZE taskset -c "$lane" nice -n 10 "$SCDEF_PY" hc.py "$ds" "$v" >>"$d/log.txt" 2>&1 \
    && log "done $ds/$v: $(tail -1 "$d/log.txt")" || log "FAILED $ds/$v; see $d/log.txt"
  return 0
}
export -f job_hc

job_score() {   # on the scDEF cores, so that the lanes' runtimes stay clean
  JAX_PLATFORMS=cpu taskset -c "$SCDEF_CPUS" nice -n 10 "$SCDEF_PY" score.py "$1" >>"$OUT/$1/score.log" 2>&1 \
    && log "scored $1" || log "FAILED score $1; see $OUT/$1/score.log"
}
export -f job_score

IDS=("$@")
if [ ${#IDS[@]} -eq 0 ]; then
  mapfile -t IDS < <("$SCDEF_PY" -c "import simcommon as S; print('\n'.join(S.all_ids()))")
fi
log "== run_all: ${#IDS[@]} data sets; scDEF jobs $N_SCDEF on cores $SCDEF_CPUS; comparator lanes:$LANES; one-layer baseline ${SIMBENCH_ONE_LAYER:-0}; pid $$"
log "GPU at start: $(nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>/dev/null | tr '\n' ';')"

# 1. reference counts, 2. simulate (the first alone: it caches splatEstimate), 3. prepare + nesting check
"$SCDEF_PY" reference.py >>"$OUT/reference.log" 2>&1 || { log "reference failed"; exit 1; }
job_simulate "${IDS[0]}"
printf '%s\n' "${IDS[@]:1}" | xargs -r -P "$N_SIM" -I{} bash -c 'job_simulate "$@"' _ {}
printf '%s\n' "${IDS[@]}" | xargs -r -P 6 -I{} bash -c 'job_prepare "$@"' _ {}
BAD=0
for ds in "${IDS[@]}"; do
  if [ ! -f "$OUT/$ds/input.h5ad" ]; then
    if grep -q "NESTING FAILED" "$OUT/$ds/prepare.log" 2>/dev/null; then BAD=1; log "NESTING FAILED: $ds"; fi
  fi
done
if [ $BAD -ne 0 ]; then log "== stopped: group levels do not nest (see <id>/nesting.json)"; exit 3; fi
READY=()
for ds in "${IDS[@]}"; do [ -f "$OUT/$ds/input.h5ad" ] && READY+=("$ds"); done
log "${#READY[@]} data sets prepared"

# 4. the two queues, in parallel
"$SCDEF_PY" -c "
import simcommon as S
for ds in '${READY[*]}'.split():
    for lb in S.gpu_jobs(ds):
        print(ds, lb)" >"$OUT/.queue_gpu"
{
  for ds in "${READY[@]}"; do
    for m in $("$SCDEF_PY" -c "import simcommon as S; print(' '.join(S.methods('$ds')))"); do echo "$ds $m"; done
  done
} >"$OUT/.queue_cpu"
log "queues: $(wc -l <"$OUT/.queue_gpu") scDEF fits, $(wc -l <"$OUT/.queue_cpu") comparator runs"
xargs -r -P "$N_SCDEF" -L1 bash -c 'job_scdef "$@"' _ <"$OUT/.queue_gpu" &
GPU_Q=$!
xargs -r -P "$N_LANES" -L1 bash -c 'job_method "$@"' _ <"$OUT/.queue_cpu" &
CPU_Q=$!
log "queues started: scDEF pid $GPU_Q, comparators pid $CPU_Q"

# 5. post stage: hierarchical-clustering baselines when ready, re-score what changed, gather, plot
post_round() {
  "$SCDEF_PY" -c "
import os, simcommon as S
def ok(path):
    r = S.read_json(path, {})
    return r.get('status') == 'ok'
for ds in '${READY[*]}'.split():
    for v, deps in S.hc_variants(ds).items():
        if S.read_json(os.path.join(S.OUT_ROOT, ds, 'methods', v, 'run.json'), {}).get('status') in ('ok', 'failed'):
            continue                    # a failed baseline is final for this run (delete its run.json to retry)
        if all(ok(os.path.join(S.OUT_ROOT, ds, 'methods', m, 'run.json')) for m in deps['methods']) and \\
           all(ok(os.path.join(S.OUT_ROOT, ds, 'scdef', f, 'fit.json')) for f in deps['fits']):
            print(ds, v)" >"$OUT/.queue_hc"
  if [ -s "$OUT/.queue_hc" ]; then
    log "baselines ready: $(wc -l <"$OUT/.queue_hc")"
    xargs -r -P "$N_LANES" -L1 bash -c 'job_hc "$@"' _ <"$OUT/.queue_hc"
  fi
  local changed=()
  for ds in "${READY[@]}"; do
    [ -n "$(find "$OUT/$ds/methods" "$OUT/$ds/scdef" -name 'run.json' -newer "$OUT/$ds/scores.csv" -o \
           -name 'fit.json' -newer "$OUT/$ds/scores.csv" 2>/dev/null | head -1)" ] && changed+=("$ds")
    [ -f "$OUT/$ds/scores.csv" ] || changed+=("$ds")
  done
  if [ ${#changed[@]} -gt 0 ]; then
    printf '%s\n' "${changed[@]}" | sort -u | xargs -r -P 2 -I{} bash -c 'job_score "$@"' _ {}
    taskset -c "$SCDEF_CPUS" "$SCDEF_PY" gather.py >>"$OUT/gather.log" 2>&1 || log "gather failed"
    taskset -c "$SCDEF_CPUS" "$SCDEF_PY" plot.py >>"$OUT/plot.log" 2>&1 || log "plot failed"
  fi
}
while :; do
  QUEUES_DONE=1
  kill -0 $GPU_Q 2>/dev/null && QUEUES_DONE=0
  kill -0 $CPU_Q 2>/dev/null && QUEUES_DONE=0
  post_round
  if [ $QUEUES_DONE -eq 1 ] && [ ! -s "$OUT/.queue_hc" ]; then break; fi
  for i in $(seq 1 60); do                      # ten minutes, or until a queue finishes
    sleep 10
    kill -0 $GPU_Q 2>/dev/null || kill -0 $CPU_Q 2>/dev/null || break
  done
done
wait $CPU_Q; wait $GPU_Q
log "status: $(tail -4 "$OUT/gather.log" | head -3 | tr '\n' ' ')"
log "== run_all finished; tables: $OUT/scores_all.csv, $OUT/level_scores_all.csv, $OUT/layer_sizes_all.csv, $OUT/batch_groups_all.csv; figures: $OUT/figures"
