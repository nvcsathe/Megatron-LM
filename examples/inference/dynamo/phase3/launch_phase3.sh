#!/usr/bin/env bash
# Host wrapper for the disaggregated Megatron stack.
#
# Required env:
#   SLURM_JOB_ID   set automatically by salloc
#   DMG_SQSH       absolute path to the dynamo-megatron container sqsh.
#                  The container must have `nixl` Python + native runtime
#                  importable; confirm with:
#                      python -c "from nixl._api import nixl_agent"
#                  inside the container before relying on this launcher.
#   STAGE          lustre staging dir holding the model checkpoint + hf-cache
#
# Optional env (passed through to orchestrate.sh):
#   MODEL_CHECKPOINT, TOKENIZER_MODEL, SERVED_MODEL_NAME, CONTEXT_LENGTH
#   TP_PREFILL, TP_DECODE                          (must be equal, default 1)
#   HTTP_PORT
#   COORD_PORT_PREFILL, COORD_PORT_DECODE          (default 5555, 5556)
#   NIXL_PORT_PREFILL, NIXL_PORT_DECODE            (default 7000, 7001)
#   MASTER_PORT_PREFILL, MASTER_PORT_DECODE        (default 29500, 29501)
#   MEGATRON_LOCAL_DEV
#   PHASE3_ASYNC_PULL_BENCHMARK=1                  run async-pull benchmark then exit
#   PHASE3_BENCH_BURSTS                            comma-separated burst sizes, default 2,4,3
#   PHASE3_BENCH_BURST_GAPS                        comma-separated seconds between bursts
#   PHASE3_BENCH_PROMPT_WORDS, PHASE3_BENCH_MAX_TOKENS, PHASE3_BENCH_WARMUP
#
# UCX transport / logging (force-set in this script, use _OVERRIDE suffix to change):
#   UCX_TLS_OVERRIDE            transport allow-list (default: cuda_ipc,cuda_copy,cma,shm,self)
#                               TCP cannot handle VRAM addresses.
#                               The NGC base image bakes in UCX_TLS=tcp; this script overrides it.
#   UCX_MEMTYPE_CACHE_OVERRIDE  set to "n" by default to avoid early-init misclassification
#   UCX_LOG_LEVEL_OVERRIDE      UCX log verbosity
#   UCX_LOG_FILE_OVERRIDE       log path; UCX expands %p to PID (default: /tmp/ucx_%p.log)
#
# If MEGATRON_LOCAL_DEV is set, the host directory it points to is mounted
# over /opt/megatron-lm so you can edit InferenceClient, coordinator, and
# disaggregation package code without rebuilding the image.

set -euo pipefail

: "${SLURM_JOB_ID:?must run inside a salloc allocation}"
: "${DMG_SQSH:?DMG_SQSH must be set (path to the dynamo-megatron sqsh on lustre)}"
: "${STAGE:?STAGE must be set (lustre staging dir)}"

[[ -f "$DMG_SQSH" ]] || { echo "DMG_SQSH not found: $DMG_SQSH" >&2; exit 1; }
[[ -d "$STAGE"   ]] || { echo "STAGE not found: $STAGE"        >&2; exit 1; }

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MEGATRON_ROOT="${MEGATRON_LOCAL_DEV:-$(cd "$SCRIPT_DIR/../../../.." && pwd)}"
[[ -d "$MEGATRON_ROOT" ]] || { echo "MEGATRON_ROOT not a dir: $MEGATRON_ROOT" >&2; exit 1; }
echo "[launch] Megatron source mount: $MEGATRON_ROOT -> /opt/megatron-lm"

MOUNTS="$STAGE:$STAGE,$MEGATRON_ROOT:/opt/megatron-lm"

# Forward env consumed by orchestrate.sh.
EXPORT_VARS="STAGE,HF_HOME,HF_TOKEN,MODEL_CHECKPOINT,MODEL_DIR,TOKENIZER_MODEL,SERVED_MODEL_NAME"
EXPORT_VARS="$EXPORT_VARS,CONTEXT_LENGTH,TP_PREFILL,TP_DECODE,PP_PREFILL,PP_DECODE,HTTP_PORT"
EXPORT_VARS="$EXPORT_VARS,COORD_PORT_PREFILL,COORD_PORT_DECODE"
EXPORT_VARS="$EXPORT_VARS,NIXL_PORT_PREFILL,NIXL_PORT_DECODE"
EXPORT_VARS="$EXPORT_VARS,MASTER_PORT_PREFILL,MASTER_PORT_DECODE"
EXPORT_VARS="$EXPORT_VARS,PHASE3_ASYNC_PULL_BENCHMARK,PHASE3_ASYNC_PULL_STRESS"
EXPORT_VARS="$EXPORT_VARS,PHASE3_BENCH_BURSTS,PHASE3_BENCH_BURST_GAPS"
EXPORT_VARS="$EXPORT_VARS,PHASE3_BENCH_PROMPT_WORDS,PHASE3_BENCH_MAX_TOKENS"
EXPORT_VARS="$EXPORT_VARS,PHASE3_BENCH_WARMUP,PHASE3_BENCH_TIMEOUT,PHASE3_BENCH_OUTPUT"

# Force CUDA-capable UCX transports for NIXL VRAM transfers.
UCX_TLS="${UCX_TLS_OVERRIDE:-cuda_ipc,cuda_copy,tcp,shm,cma,self}"
UCX_MEMTYPE_CACHE="${UCX_MEMTYPE_CACHE_OVERRIDE:-n}"
UCX_LOG_LEVEL="${UCX_LOG_LEVEL_OVERRIDE:-info}"
UCX_LOG_FILE="${UCX_LOG_FILE_OVERRIDE:-/tmp/ucx_%p.log}"
export UCX_TLS UCX_MEMTYPE_CACHE UCX_LOG_LEVEL UCX_LOG_FILE

echo "[launch] container: $DMG_SQSH"
echo "[launch] mounts:    $MOUNTS"
echo "[launch] UCX_TLS=$UCX_TLS  UCX_LOG_FILE=$UCX_LOG_FILE"
echo "[launch] expect 'PHASE3_READY' on stdout when ready"
RUN_ASYNC_PULL_BENCHMARK="${PHASE3_ASYNC_PULL_BENCHMARK:-${PHASE3_ASYNC_PULL_STRESS:-0}}"
if [[ "$RUN_ASYNC_PULL_BENCHMARK" == "1" ]]; then
    echo "[launch] async NIXL pull benchmark enabled"
fi
echo

if [[ "$RUN_ASYNC_PULL_BENCHMARK" == "1" ]]; then
    PHASE3_ENTRYPOINT='
set -euo pipefail
rm -f /tmp/phase3.env
bash /opt/megatron-lm/examples/inference/dynamo/phase3/orchestrate.sh &
orch_pid=$!
cleanup() {
    kill "$orch_pid" 2>/dev/null || true
    wait "$orch_pid" 2>/dev/null || true
}
trap cleanup EXIT

ready=0
for _ in $(seq 1 600); do
    if [[ -f /tmp/phase3.env ]]; then
        # shellcheck disable=SC1091
        source /tmp/phase3.env
        if curl -sf "$PHASE3_FRONTEND_URL/v1/models" | grep -q "$PHASE3_MODEL_NAME"; then
            ready=1
            break
        fi
    fi
    if ! kill -0 "$orch_pid" 2>/dev/null; then
        echo "[launch] orchestrate.sh exited before readiness" >&2
        wait "$orch_pid"
    fi
    sleep 1
done

if [[ "$ready" != "1" ]]; then
    echo "[launch] timed out waiting for stack readiness" >&2
    exit 1
fi

# shellcheck disable=SC1091
source /tmp/phase3.env
bench_status=0
python /opt/megatron-lm/examples/inference/dynamo/phase3/benchmark_async_pull.py || bench_status=$?
if ! kill -0 "$orch_pid" 2>/dev/null; then
    echo "[launch] orchestrate.sh exited during benchmark" >&2
    wait "$orch_pid" || true
fi
exit "$bench_status"
'
else
    PHASE3_ENTRYPOINT='exec bash /opt/megatron-lm/examples/inference/dynamo/phase3/orchestrate.sh'
fi

exec srun \
    --jobid="$SLURM_JOB_ID" --overlap \
    --container-image="$DMG_SQSH" \
    --container-name=dmg \
    --container-mounts="$MOUNTS" \
    --container-workdir=/opt/megatron-lm \
    --export="ALL,$EXPORT_VARS,UCX_TLS=$UCX_TLS,UCX_MEMTYPE_CACHE=$UCX_MEMTYPE_CACHE,UCX_LOG_LEVEL=$UCX_LOG_LEVEL,UCX_LOG_FILE=$UCX_LOG_FILE" \
    bash -lc "$PHASE3_ENTRYPOINT"
