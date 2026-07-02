#!/usr/bin/env bash
# Launch matched TP=1/PP=1 Nano v3 prefill and decode workers on one Slurm node.

set -uo pipefail

export UCX_TLS="${UCX_TLS_OVERRIDE:-cuda_ipc,cuda_copy,tcp,shm,cma,self}"
export UCX_MEMTYPE_CACHE="${UCX_MEMTYPE_CACHE_OVERRIDE:-n}"
export UCX_LOG_LEVEL="${UCX_LOG_LEVEL_OVERRIDE:-info}"
export UCX_LOG_FILE="${UCX_LOG_FILE_OVERRIDE:-/tmp/ucx_%p.log}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"

STAGE="${STAGE:-/lustre/fsw/portfolios/nemotron/users/csathe}"
# Cluster artifact defaults; launch.sh mounts them at these paths.
MODEL_CHECKPOINT="${MODEL_CHECKPOINT:-/lustre/fsw/portfolios/llmservice/users/ksanthanam/nemotron-3-nano-30b}"
PRETRAINED_CHECKPOINT="${PRETRAINED_CHECKPOINT:-/lustre/fsw/portfolios/llmservice/users/ksanthanam/nanov3}"
TOKENIZER_MODEL="${TOKENIZER_MODEL:-/lustre/fsw/portfolios/llmservice/projects/llmservice_nlp_fm/nemotron6/tokenizers/multiMixV8.gpt4o_nc_sd.500000.128k.vocab.json}"
DYNAMO_MODEL="${DYNAMO_MODEL:-nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-nemotron3-nano}"
PREFLIGHT_ONLY="${PREFLIGHT_ONLY:-0}"
CONTEXT_LENGTH="${CONTEXT_LENGTH:-4096}"
INFER_MAX_SEQ_LEN="${INFER_MAX_SEQ_LEN:-$CONTEXT_LENGTH}"
INFER_BUFFER_GB="${INFER_BUFFER_GB:-20}"
INFER_MAX_TOKENS="${INFER_MAX_TOKENS:-8192}"
INFER_MAX_REQUESTS="${INFER_MAX_REQUESTS:-256}"

PREFIX_CACHE="${PREFIX_CACHE:-1}"
MAMBA_GB="${MAMBA_GB:-4.0}"

# The optional baseline needs a separate EP-sized GPU set.
WITH_BASELINE="${WITH_BASELINE:-0}"

ROLE_EP_SIZE="${ROLE_EP_SIZE:-2}"
GPU_PREFILL="${GPU_PREFILL:-0,1}"
GPU_DECODE="${GPU_DECODE:-2,3}"
GPU_BASELINE="${GPU_BASELINE:-4,5}"

HTTP_PORT="${HTTP_PORT:-8100}"
HTTP_PORT_AGG="${HTTP_PORT_AGG:-8101}"
COORD_PORT_PREFILL="${COORD_PORT_PREFILL:-5555}"
COORD_PORT_DECODE="${COORD_PORT_DECODE:-5556}"
COORD_PORT_AGG="${COORD_PORT_AGG:-5557}"
NIXL_PORT_PREFILL="${NIXL_PORT_PREFILL:-7000}"
NIXL_PORT_DECODE="${NIXL_PORT_DECODE:-7001}"
MASTER_PORT_PREFILL="${MASTER_PORT_PREFILL:-29500}"
MASTER_PORT_DECODE="${MASTER_PORT_DECODE:-29501}"
MASTER_PORT_AGG="${MASTER_PORT_AGG:-29502}"

export NATS_SERVER="nats://127.0.0.1:4222"
export ETCD_ENDPOINTS="http://127.0.0.1:2379"
export HF_HOME="${HF_HOME:-${STAGE}/hf-cache}"
export CUDA_DEVICE_MAX_CONNECTIONS=1
export MASTER_ADDR=127.0.0.1

LOG_DIR="${LOG_DIR:-/tmp}"
PIDS=()

log()  { printf '[orchestrate %s] %s\n' "$(date +%H:%M:%S)" "$*"; }
die()  { log "FATAL: $*" >&2; cleanup; exit 1; }

cleanup() {
    log "cleaning up..."
    for pid in "${PIDS[@]-}"; do
        [[ -n "$pid" ]] && kill "$pid" 2>/dev/null || true
    done
    wait 2>/dev/null || true
}
trap cleanup EXIT INT TERM

wait_for() {
    local desc="$1" max="$2"; shift 2
    local elapsed=0
    while ! "$@" >/dev/null 2>&1; do
        sleep 2; elapsed=$((elapsed + 2))
        if [[ $elapsed -ge $max ]]; then log "TIMEOUT waiting for $desc after ${max}s"; return 1; fi
    done
    log "ready: $desc (${elapsed}s)"
}

count_csv() {
    local csv="$1"
    local -a parts
    IFS=',' read -ra parts <<< "$csv"
    echo "${#parts[@]}"
}

require_gpu_count() {
    local name="$1" csv="$2" expected="$3"
    local count
    count=$(count_csv "$csv")
    [[ "$count" == "$expected" ]] || die "$name must contain exactly $expected GPU ids for EP=$expected (got '$csv', count=$count)"
}

require_gpu_count GPU_PREFILL "$GPU_PREFILL" "$ROLE_EP_SIZE"
require_gpu_count GPU_DECODE "$GPU_DECODE" "$ROLE_EP_SIZE"
if [[ "$WITH_BASELINE" == "1" ]]; then
    require_gpu_count GPU_BASELINE "$GPU_BASELINE" "$ROLE_EP_SIZE"
fi
if [[ -e "$DYNAMO_MODEL" && ! -d "$DYNAMO_MODEL" ]]; then
    die "DYNAMO_MODEL must be a directory or HF model id for Dynamo registration (got file: $DYNAMO_MODEL)"
fi
[[ -f "$TOKENIZER_MODEL" ]] \
    || die "Megatron TOKENIZER_MODEL is not visible in the container: $TOKENIZER_MODEL (set TOKENIZER_MODEL to the checkpoint vocab and mount its parent directory)"

log "Hybrid disagg (Mamba transfer): TP=1 PP=1 EP=$ROLE_EP_SIZE; prefill GPUs=$GPU_PREFILL, decode GPUs=$GPU_DECODE; baseline=$WITH_BASELINE (GPUs=$GPU_BASELINE)"

# Resolve Dynamo tokenizer metadata without downloading model weights.
resolve_dynamo_metadata() {
    if [[ -d "$DYNAMO_MODEL" ]]; then
        printf '%s\n' "$DYNAMO_MODEL"
        return 0
    fi

    python -c \
        'import asyncio, sys
from dynamo.llm import fetch_model
async def main():
    return await fetch_model(sys.argv[1], ignore_weights=True)
print(asyncio.run(main()))' \
        "$DYNAMO_MODEL"
}

log "resolving Dynamo tokenizer metadata for $DYNAMO_MODEL (weights excluded)..."
DYNAMO_MODEL_METADATA=$(resolve_dynamo_metadata) \
    || die "could not resolve Dynamo metadata for '$DYNAMO_MODEL' (check HF_HOME/network/HF_TOKEN)"
DYNAMO_MODEL_METADATA="${DYNAMO_MODEL_METADATA##*$'\n'}"
[[ -d "$DYNAMO_MODEL_METADATA" ]] \
    || die "Dynamo metadata resolver returned a non-directory: '$DYNAMO_MODEL_METADATA'"
[[ -f "$DYNAMO_MODEL_METADATA/config.json" ]] \
    || die "Dynamo metadata is missing config.json: $DYNAMO_MODEL_METADATA"
[[ -f "$DYNAMO_MODEL_METADATA/tokenizer.json" ]] \
    || die "Dynamo metadata is missing tokenizer.json: $DYNAMO_MODEL_METADATA (a bare Megatron vocab.json is not sufficient)"
log "Dynamo tokenizer metadata ready: $DYNAMO_MODEL_METADATA"

if [[ "$PREFLIGHT_ONLY" == "1" ]]; then
    echo "NANO_V3_PREFLIGHT_OK"
    exit 0
fi

# Nano v3 serving arguments. MODEL_ARGS_OVERRIDE replaces this list.
if [[ -n "${MODEL_ARGS_OVERRIDE:-}" ]]; then
    # shellcheck disable=SC2206
    MODEL_ARGS=( $MODEL_ARGS_OVERRIDE )
else
    MODEL_ARGS=(
        --model-provider hybrid
        --pretrained-checkpoint "$PRETRAINED_CHECKPOINT"
        --use-checkpoint-args
        --dist-ckpt-strictness log_unexpected
        --bf16
        --sequence-parallel
        --expert-tensor-parallel-size 1
        --attention-backend flash
        --moe-router-score-function sigmoid
        --moe-router-enable-expert-bias
        --moe-router-topk-scaling-factor 2.5
        --moe-token-dispatcher-type alltoall
        --moe-grouped-gemm
        --moe-router-dtype fp32
        --moe-shared-expert-overlap
        --seq-length 73728
        --max-position-embeddings 73728
        --inference-max-seq-length "$INFER_MAX_SEQ_LEN"
        --transformer-impl inference_optimized
        --te-rng-tracker
        --inference-rng-tracker
        --cuda-graph-impl local
        --inference-grouped-gemm-backend vllm
        --inference-use-synchronous-zmq-collectives
        --inference-dynamic-batching-buffer-size-gb "$INFER_BUFFER_GB"
        --inference-dynamic-batching-max-tokens "$INFER_MAX_TOKENS"
        --enable-chunked-prefill
        --inference-dynamic-batching-num-cuda-graphs -1
        --inference-cuda-graph-scope block
        --inference-dynamic-batching-max-requests "$INFER_MAX_REQUESTS"
        --inference-logging-step-interval 100
        --micro-batch-size 1
    )
fi

PREFIX_ARGS=()
if [[ "$PREFIX_CACHE" == "1" ]]; then
    PREFIX_ARGS=(
        --inference-dynamic-batching-prefix-caching
        --inference-dynamic-batching-prefix-caching-mamba-gb "$MAMBA_GB"
    )
fi
TOKENIZER_ARGS=()
TOKENIZER_ARGS=(--tokenizer-model "$TOKENIZER_MODEL")

# Args: role, GPUs, coordinator port, NIXL port, log, extra Megatron args.
launch_engine() {
    local role="$1" gpus="$2" coord_port="$3" nixl_port="$4" logf="$5"; shift 5
    local nproc="$ROLE_EP_SIZE"
    local transfer_args=()
    if [[ "$role" != "aggregated" ]]; then
        transfer_args=(--kv-transfer-listen-addr "127.0.0.1:$nixl_port")
    fi
    log "starting owned Megatron $role engine (GPUs=$gpus, TP=1 PP=1 EP=$ROLE_EP_SIZE)..."
    (
        CUDA_VISIBLE_DEVICES="$gpus" exec python -m megatron.inference.integrations.dynamo \
                --role "$role" \
                --model "$DYNAMO_MODEL_METADATA" \
                --served-model-name "$SERVED_MODEL_NAME" \
                --nproc-per-node "$nproc" \
                --coordinator-host 127.0.0.1 \
                --coordinator-port "$coord_port" \
                "${transfer_args[@]}" \
                --megatron-root /opt/megatron-lm \
                -- \
                --tensor-model-parallel-size 1 \
                --pipeline-model-parallel-size 1 \
                --expert-model-parallel-size "$ROLE_EP_SIZE" \
                --load "$MODEL_CHECKPOINT" \
                "${TOKENIZER_ARGS[@]}" \
                "${MODEL_ARGS[@]}" "$@"
    ) > "$logf" 2>&1 &
    PIDS+=($!)
}

# Runtime services.
log "starting NATS..."
nats-server --jetstream --store_dir /tmp/nats-jetstream --port 4222 -m 8222 \
    > "$LOG_DIR/nats.log" 2>&1 &
PIDS+=($!)
log "starting etcd..."
etcd --data-dir /tmp/etcd-data \
     --listen-client-urls http://0.0.0.0:2379 \
     --advertise-client-urls http://0.0.0.0:2379 \
     > "$LOG_DIR/etcd.log" 2>&1 &
PIDS+=($!)
wait_for "nats /healthz"  30 curl -sf http://127.0.0.1:8222/healthz || die "nats never healthy"
wait_for "etcd /health"   30 curl -sf http://127.0.0.1:2379/health  || die "etcd never healthy"

# Start both workers concurrently; model loading may take several minutes.
launch_engine prefill "$GPU_PREFILL" "$COORD_PORT_PREFILL" "$NIXL_PORT_PREFILL" \
    "$LOG_DIR/worker-prefill.log" \
    "${PREFIX_ARGS[@]}"

launch_engine decode "$GPU_DECODE" "$COORD_PORT_DECODE" "$NIXL_PORT_DECODE" \
    "$LOG_DIR/worker-decode.log" \
    "${PREFIX_ARGS[@]}"

wait_for "prefill worker registered" 1800 \
    grep -Eq "Registered base model|Starting NATS push endpoint listener" \
        "$LOG_DIR/worker-prefill.log" \
    || die "prefill worker never registered (see $LOG_DIR/worker-prefill.log)"
wait_for "decode worker registered" 1800 \
    grep -Eq "Registered base model|Starting NATS push endpoint listener" \
        "$LOG_DIR/worker-decode.log" \
    || die "decode worker never registered (see $LOG_DIR/worker-decode.log)"

log "starting Dynamo frontend (disagg) on :$HTTP_PORT..."
python -m dynamo.frontend --http-port "$HTTP_PORT" --router-mode kv \
    --request-plane nats --event-plane nats > "$LOG_DIR/frontend.log" 2>&1 &
PIDS+=($!)
wait_for "frontend exposes $SERVED_MODEL_NAME" 60 \
    bash -c "curl -sf http://127.0.0.1:$HTTP_PORT/v1/models | grep -q '$SERVED_MODEL_NAME'" \
    || die "frontend never exposed model (see $LOG_DIR/frontend.log)"

# Optional aggregated reference.
BASELINE_URL=""
if [[ "$WITH_BASELINE" == "1" ]]; then
    DYN_NAMESPACE=baseline launch_engine aggregated "$GPU_BASELINE" "$COORD_PORT_AGG" "" \
        "$LOG_DIR/worker-agg.log"
    wait_for "agg worker registered" 900 \
        grep -Eq "Registered base model|Starting NATS push endpoint listener" \
            "$LOG_DIR/worker-agg.log" \
        || die "agg worker never registered (see $LOG_DIR/worker-agg.log)"

    log "starting Dynamo frontend (baseline) on :$HTTP_PORT_AGG..."
    DYN_NAMESPACE=baseline python -m dynamo.frontend --http-port "$HTTP_PORT_AGG" --router-mode kv \
        --request-plane nats --event-plane nats > "$LOG_DIR/frontend-agg.log" 2>&1 &
    PIDS+=($!)
    wait_for "baseline frontend exposes $SERVED_MODEL_NAME" 60 \
        bash -c "curl -sf http://127.0.0.1:$HTTP_PORT_AGG/v1/models | grep -q '$SERVED_MODEL_NAME'" \
        || die "baseline frontend never exposed model (see $LOG_DIR/frontend-agg.log)"
    BASELINE_URL="http://127.0.0.1:$HTTP_PORT_AGG"
fi

# Publish connection details for verify.sh.
cat > /tmp/nano_v3_test.env <<ENV
export NANO_V3_FRONTEND_URL="http://127.0.0.1:$HTTP_PORT"
export NANO_V3_BASELINE_URL="$BASELINE_URL"
export NANO_V3_MODEL_NAME="$SERVED_MODEL_NAME"
export NANO_V3_LOG_DIR="$LOG_DIR"
export NANO_V3_PREFILL_LOG="$LOG_DIR/worker-prefill.log"
export NANO_V3_DECODE_LOG="$LOG_DIR/worker-decode.log"
ENV

log "all components healthy."
log "  disagg:   http://127.0.0.1:$HTTP_PORT"
[[ -n "$BASELINE_URL" ]] && log "  baseline: $BASELINE_URL"
echo "NANO_V3_TEST_READY"
wait -n "${PIDS[@]}"
log "a component exited; tearing down"
exit 1
