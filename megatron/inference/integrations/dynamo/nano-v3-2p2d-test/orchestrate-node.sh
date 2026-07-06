#!/usr/bin/env bash
# Run on every selected node. Node zero owns shared services and the workload.

set -uo pipefail

NODE_RANK="${SLURM_NODEID:-0}"
IFS=',' read -ra TEST_HOSTS <<< "$TEST_HOSTS_CSV"
NODE_HOST="${TEST_HOSTS[$NODE_RANK]}"
WORKERS_PER_NODE=$((GPUS_PER_NODE / ROLE_EP_SIZE))
TOTAL_WORKERS=$((PREFILL_WORKERS + DECODE_WORKERS))

NAMESPACE="${NAMESPACE:-nano-2p2d}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-nemotron3-nano}"
CONTEXT_LENGTH="${CONTEXT_LENGTH:-8192}"
INFER_MAX_SEQ_LEN="${INFER_MAX_SEQ_LEN:-$CONTEXT_LENGTH}"
INFER_BUFFER_GB="${INFER_BUFFER_GB:-20}"
INFER_MAX_TOKENS="${INFER_MAX_TOKENS:-8192}"
INFER_MAX_REQUESTS="${INFER_MAX_REQUESTS:-16}"
KV_BLOCK_SIZE="${KV_BLOCK_SIZE:-256}"
MAMBA_GB="${MAMBA_GB:-4.0}"
PREFIX_CACHE_EVICTION_POLICY="${PREFIX_CACHE_EVICTION_POLICY:-lru}"
CUDA_GRAPH_IMPL="${CUDA_GRAPH_IMPL:-none}"

HTTP_PORT="${HTTP_PORT:-8100}"
NATS_PORT="${NATS_PORT:-4222}"
NATS_MONITOR_PORT="${NATS_MONITOR_PORT:-8222}"
ETCD_PORT="${ETCD_PORT:-2379}"
COORD_PORT_BASE="${COORD_PORT_BASE:-5555}"
NIXL_PORT_BASE="${NIXL_PORT_BASE:-7000}"
WORKER_START_TIMEOUT="${WORKER_START_TIMEOUT:-1800}"
TEST_TIMEOUT_SECONDS="${TEST_TIMEOUT_SECONDS:-3600}"

export NATS_SERVER="nats://$HEAD_HOST:$NATS_PORT"
export ETCD_ENDPOINTS="http://$HEAD_HOST:$ETCD_PORT"
export HF_HOME="${HF_HOME:-$STAGE/hf-cache}"
export PYTHONUNBUFFERED=1
export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-1}"
export MAMBA_DETERMINISTIC="${MAMBA_DETERMINISTIC:-1}"
export TRITON_CACHE_AUTOTUNING="${TRITON_CACHE_AUTOTUNING:-0}"
export UCX_TLS="${UCX_TLS_OVERRIDE:-cuda_ipc,cuda_copy,tcp,shm,cma,self}"
export UCX_MEMTYPE_CACHE="${UCX_MEMTYPE_CACHE_OVERRIDE:-n}"

LOG_DIR="$RUN_DIR/logs"
READY_DIR="$RUN_DIR/ready"
mkdir -p "$LOG_DIR" "$READY_DIR"

VISIBLE_GPU_COUNT="$(nvidia-smi -L | wc -l)"
if (( VISIBLE_GPU_COUNT < GPUS_PER_NODE )); then
    echo "node $NODE_RANK exposes $VISIBLE_GPU_COUNT GPUs, expected $GPUS_PER_NODE" >&2
    exit 1
fi

PIDS=()
TEST_RC=0
log() { printf '[node %s %s] %s\n' "$NODE_RANK" "$(date +%H:%M:%S)" "$*"; }
cleanup() {
    if (( NODE_RANK == 0 )) && [[ ! -e "$RUN_DIR/test-complete" ]]; then
        printf '%s\n' 1 > "$RUN_DIR/test-exit-code"
        : > "$RUN_DIR/test-complete"
    fi
    for pid in "${PIDS[@]-}"; do [[ -n "$pid" ]] && kill "$pid" 2>/dev/null || true; done
    wait 2>/dev/null || true
}
trap cleanup EXIT INT TERM

wait_for() {
    local description="$1" timeout="$2"; shift 2
    local elapsed=0
    while ! "$@" >/dev/null 2>&1; do
        sleep 2; elapsed=$((elapsed + 2))
        if (( elapsed >= timeout )); then
            log "timeout waiting for $description after ${timeout}s"
            return 1
        fi
    done
}

if (( NODE_RANK == 0 )); then
    nats-server --jetstream --store_dir "$RUN_DIR/nats-jetstream" \
        --addr 0.0.0.0 --port "$NATS_PORT" -m "$NATS_MONITOR_PORT" \
        > "$LOG_DIR/nats.log" 2>&1 &
    PIDS+=("$!")
    etcd --data-dir "$RUN_DIR/etcd-data" \
        --listen-client-urls "http://0.0.0.0:$ETCD_PORT" \
        --advertise-client-urls "http://$HEAD_HOST:$ETCD_PORT" \
        > "$LOG_DIR/etcd.log" 2>&1 &
    PIDS+=("$!")
fi
wait_for "NATS" 60 curl -sf "http://$HEAD_HOST:$NATS_MONITOR_PORT/healthz" || exit 1
wait_for "etcd" 60 curl -sf "http://$HEAD_HOST:$ETCD_PORT/health" || exit 1

MODEL_PATH_FILE="$RUN_DIR/dynamo-model-path"
if (( NODE_RANK == 0 )); then
    log "resolving Dynamo model metadata"
    python -c \
        'import asyncio, pathlib, sys
from dynamo.llm import fetch_model
model, output = sys.argv[1:]
async def resolve(): return await fetch_model(model, ignore_weights=True)
path = model if pathlib.Path(model).is_dir() else asyncio.run(resolve())
pathlib.Path(output).write_text(str(path))' \
        "$DYNAMO_MODEL" "$MODEL_PATH_FILE" || exit 1
fi
wait_for "Dynamo model metadata" 300 test -s "$MODEL_PATH_FILE" || exit 1
DYNAMO_MODEL_METADATA="$(<"$MODEL_PATH_FILE")"
for metadata_file in config.json tokenizer.json; do
    [[ -f "$DYNAMO_MODEL_METADATA/$metadata_file" ]] || {
        log "Dynamo metadata is missing $metadata_file: $DYNAMO_MODEL_METADATA"; exit 1;
    }
done

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
        --cuda-graph-impl "$CUDA_GRAPH_IMPL"
        --inference-grouped-gemm-backend vllm
        --inference-use-synchronous-zmq-collectives
        --inference-dynamic-batching-buffer-size-gb "$INFER_BUFFER_GB"
        --inference-dynamic-batching-max-tokens "$INFER_MAX_TOKENS"
        --inference-dynamic-batching-block-size "$KV_BLOCK_SIZE"
        --enable-chunked-prefill
        --inference-dynamic-batching-max-requests "$INFER_MAX_REQUESTS"
        --inference-logging-step-interval 1
        --micro-batch-size 1
    )
    if [[ "$CUDA_GRAPH_IMPL" == "local" ]]; then
        MODEL_ARGS+=(
            --inference-dynamic-batching-num-cuda-graphs -1
            --inference-cuda-graph-scope block
        )
    elif [[ "$CUDA_GRAPH_IMPL" != "none" ]]; then
        log "CUDA_GRAPH_IMPL must be none or local, got $CUDA_GRAPH_IMPL"
        exit 2
    fi
fi
PREFIX_ARGS=(
    --inference-dynamic-batching-prefix-caching
    --inference-dynamic-batching-prefix-caching-eviction-policy "$PREFIX_CACHE_EVICTION_POLICY"
    --inference-dynamic-batching-prefix-caching-mamba-gb "$MAMBA_GB"
)

for ((local_worker = 0; local_worker < WORKERS_PER_NODE; local_worker++)); do
    global_worker=$((NODE_RANK * WORKERS_PER_NODE + local_worker))
    if (( global_worker < PREFILL_WORKERS )); then
        role=prefill
        component=prefill
        role_index=$global_worker
    else
        role=decode
        component=backend
        role_index=$((global_worker - PREFILL_WORKERS))
    fi
    first_gpu=$((local_worker * ROLE_EP_SIZE))
    gpu_list="$first_gpu,$((first_gpu + 1))"
    worker_log="$LOG_DIR/node-${NODE_RANK}-worker-${global_worker}-${role}.log"
    worker_id_file="$READY_DIR/worker-$global_worker"
    : > "$worker_log"
    log "starting $role worker $role_index on GPUs $gpu_list"
    (
        CUDA_VISIBLE_DEVICES="$gpu_list" exec python -m megatron.inference.integrations.dynamo \
            --role "$role" \
            --namespace "$NAMESPACE" \
            --component "$component" \
            --model "$DYNAMO_MODEL_METADATA" \
            --served-model-name "$SERVED_MODEL_NAME" \
            --nproc-per-node "$ROLE_EP_SIZE" \
            --coordinator-host "$NODE_HOST" \
            --coordinator-port "$((COORD_PORT_BASE + local_worker))" \
            --kv-transfer-listen-addr "$NODE_HOST:$((NIXL_PORT_BASE + local_worker))" \
            --worker-id-file "$worker_id_file" \
            --engine-start-timeout "$WORKER_START_TIMEOUT" \
            --megatron-root /opt/megatron-lm \
            -- \
            --tensor-model-parallel-size 1 \
            --pipeline-model-parallel-size 1 \
            --expert-model-parallel-size "$ROLE_EP_SIZE" \
            --load "$MODEL_CHECKPOINT" \
            --tokenizer-model "$TOKENIZER_MODEL" \
            "${MODEL_ARGS[@]}" \
            "${PREFIX_ARGS[@]}"
    ) > "$worker_log" 2>&1 &
    PIDS+=("$!")
done

for ((local_worker = 0; local_worker < WORKERS_PER_NODE; local_worker++)); do
    global_worker=$((NODE_RANK * WORKERS_PER_NODE + local_worker))
    worker_log="$(find "$LOG_DIR" -maxdepth 1 -name "node-${NODE_RANK}-worker-${global_worker}-*.log" -print -quit)"
    wait_for "worker $global_worker registration" "$WORKER_START_TIMEOUT" \
        grep -Eq "Registered base model|Starting NATS push endpoint listener" "$worker_log" \
        || { tail -n 100 "$worker_log" >&2; exit 1; }
    wait_for "worker $global_worker identity" "$WORKER_START_TIMEOUT" \
        test -s "$READY_DIR/worker-$global_worker" || exit 1
done

if (( NODE_RANK == 0 )); then
    wait_for "$TOTAL_WORKERS worker identities" "$WORKER_START_TIMEOUT" \
        bash -c "test \"\$(find '$READY_DIR' -type f -name 'worker-*' | wc -l)\" -eq '$TOTAL_WORKERS'" \
        || exit 1

    log "starting KV-aware disaggregated frontend"
    python -m dynamo.frontend \
        --namespace "$NAMESPACE" \
        --http-port "$HTTP_PORT" \
        --router-mode kv \
        --request-plane nats \
        --event-plane nats > "$LOG_DIR/frontend.log" 2>&1 &
    PIDS+=("$!")
    wait_for "frontend model" 120 \
        bash -c "curl -sf 'http://127.0.0.1:$HTTP_PORT/v1/models' | grep -q '$SERVED_MODEL_NAME'" \
        || { tail -n 100 "$LOG_DIR/frontend.log" >&2; exit 1; }
    sleep "${FRONTEND_SETTLE_SECONDS:-5}"

    log "running 2P2D workload"
    timeout "$TEST_TIMEOUT_SECONDS" python \
        /opt/megatron-lm/megatron/inference/integrations/dynamo/nano-v3-2p2d-test/workload.py \
        --url "http://127.0.0.1:$HTTP_PORT" \
        --model "$SERVED_MODEL_NAME" \
        --run-dir "$RUN_DIR" \
        --expected-prefill "$PREFILL_WORKERS" \
        --expected-decode "$DECODE_WORKERS" \
        --concurrency "${REQUEST_CONCURRENCY:-4}" \
        --max-requests "${MAX_REQUESTS:-32}" \
        --max-tokens "${MAX_TOKENS:-32}" \
        --prompt-repeat "${PROMPT_REPEAT:-1024}" \
        --log-settle-seconds "${LOG_SETTLE_SECONDS:-3}" \
        2>&1 | tee "$LOG_DIR/workload.log"
    TEST_RC=${PIPESTATUS[0]}
    printf '%s\n' "$TEST_RC" > "$RUN_DIR/test-exit-code"
    : > "$RUN_DIR/test-complete"
else
    wait_for "test completion" "$((WORKER_START_TIMEOUT + TEST_TIMEOUT_SECONDS))" \
        test -f "$RUN_DIR/test-complete" || TEST_RC=1
fi

exit "$TEST_RC"

