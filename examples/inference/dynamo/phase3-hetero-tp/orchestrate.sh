#!/usr/bin/env bash
# Heterogeneous-TP/PP disagg orchestrator. Identical to phase3/orchestrate.sh
# EXCEPT it allows TP_PREFILL != TP_DECODE and PP_PREFILL != PP_DECODE. The KV
# handoff re-shards KV heads across the TP boundary (see
# megatron/core/inference/disaggregation/kv_reshard.py and
# disaggregation/transfer_backends/nixl.py).
#
#     NATS + etcd
#     ├── Megatron coordinator [PREFILL]  (torchrun TP=$TP_PREFILL PP=$PP_PREFILL, GPUs 0..P-1)
#     ├── Megatron coordinator [DECODE]   (torchrun TP=$TP_DECODE  PP=$PP_DECODE,  GPUs P..P+D-1)
#     ├── megatron.inference.integrations.dynamo worker [PREFILL] --role prefill
#     ├── megatron.inference.integrations.dynamo worker [DECODE]  --role decode
#     └── dynamo.frontend
#
# Each coordinator claims TP*PP GPUs; prefill takes the first
# TP_PREFILL*PP_PREFILL devices and decode the next TP_DECODE*PP_DECODE.
#
# Constraint: one of {TP_PREFILL, TP_DECODE} must divide the other, and both
# must divide num_query_groups (KV heads). For Llama-3.1-8B num_query_groups=8,
# so valid TP pairs are drawn from {1,2,4,8}. Past TP=8 KV heads replicate and
# re-shard is unsupported (use matched TP there). PP just changes how the 32
# layers are split per coordinator; it must divide num-layers (32).
#
# Outputs: /tmp/phase3_hetero.env + the usual per-component logs in $LOG_DIR.
# Emits "PHASE3_HETERO_READY" on stdout once healthy.

set -uo pipefail

export UCX_TLS="${UCX_TLS_OVERRIDE:-cuda_ipc,cuda_copy,tcp,shm,cma,self}"
export UCX_MEMTYPE_CACHE="${UCX_MEMTYPE_CACHE_OVERRIDE:-n}"
export UCX_LOG_LEVEL="${UCX_LOG_LEVEL_OVERRIDE:-info}"
export UCX_LOG_FILE="${UCX_LOG_FILE_OVERRIDE:-/tmp/ucx_%p.log}"

STAGE="${STAGE:-/lustre/fsw/portfolios/nemotron/users/csathe}"
MODEL_DIR="${MODEL_DIR:-llama3.1-8b-mcore}"
MODEL_CHECKPOINT="${MODEL_CHECKPOINT:-$STAGE/models/${MODEL_DIR}}"
TOKENIZER_MODEL="${TOKENIZER_MODEL:-meta-llama/Llama-3.1-8B-Instruct}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-llama-3.1-8b-instruct}"
CONTEXT_LENGTH="${CONTEXT_LENGTH:-4096}"
NUM_QUERY_GROUPS="${NUM_QUERY_GROUPS:-8}"

# Heterogeneous disagg topology — the whole point of this script.
TP_PREFILL="${TP_PREFILL:-2}"
TP_DECODE="${TP_DECODE:-4}"
PP_PREFILL="${PP_PREFILL:-1}"
PP_DECODE="${PP_DECODE:-1}"
NUM_LAYERS="${NUM_LAYERS:-32}"

# Validate compatibility up-front so we fail fast with a clear message rather
# than deep inside reshard_plan at the first request.
if (( TP_PREFILL % TP_DECODE != 0 && TP_DECODE % TP_PREFILL != 0 )); then
    echo "FATAL: one of TP_PREFILL($TP_PREFILL)/TP_DECODE($TP_DECODE) must divide the other" >&2
    exit 1
fi
if (( NUM_QUERY_GROUPS % TP_PREFILL != 0 || NUM_QUERY_GROUPS % TP_DECODE != 0 )); then
    echo "FATAL: both TP sizes must divide num_query_groups($NUM_QUERY_GROUPS); past that KV heads replicate (unsupported)" >&2
    exit 1
fi
if (( NUM_LAYERS % PP_PREFILL != 0 || NUM_LAYERS % PP_DECODE != 0 )); then
    echo "FATAL: both PP sizes must divide num-layers($NUM_LAYERS)" >&2
    exit 1
fi

# World size per coordinator = TP*PP; each rank gets its own GPU.
WORLD_PREFILL=$((TP_PREFILL * PP_PREFILL))
WORLD_DECODE=$((TP_DECODE * PP_DECODE))

HTTP_PORT="${HTTP_PORT:-8100}"
COORD_PORT_PREFILL="${COORD_PORT_PREFILL:-5555}"
COORD_PORT_DECODE="${COORD_PORT_DECODE:-5556}"
NIXL_PORT_PREFILL="${NIXL_PORT_PREFILL:-7000}"
NIXL_PORT_DECODE="${NIXL_PORT_DECODE:-7001}"
MASTER_PORT_PREFILL="${MASTER_PORT_PREFILL:-29500}"
MASTER_PORT_DECODE="${MASTER_PORT_DECODE:-29501}"

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
    for pid in "${PIDS[@]}"; do kill "$pid" 2>/dev/null || true; done
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

log "Heterogeneous topology: prefill TP=$TP_PREFILL PP=$PP_PREFILL ($WORLD_PREFILL GPUs), decode TP=$TP_DECODE PP=$PP_DECODE ($WORLD_DECODE GPUs); num_query_groups=$NUM_QUERY_GROUPS num_layers=$NUM_LAYERS"

###############################################################################
# 1. NATS + etcd
###############################################################################
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

###############################################################################
# 2. Two Megatron coordinators (prefill + decode) at DIFFERENT TP
###############################################################################
MODEL_ARGS=(
    --ckpt-format torch_dist
    --use-checkpoint-args
    --disable-bias-linear
    --transformer-impl transformer_engine
    --normalization RMSNorm
    --group-query-attention --num-query-groups "$NUM_QUERY_GROUPS"
    --no-masked-softmax-fusion
    --attention-softmax-in-fp32
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --untie-embeddings-and-output-weights
    --position-embedding-type rope
    --rotary-percent 1.0
    --rotary-base 500000
    --use-rope-scaling
    --use-rotary-position-embeddings
    --swiglu
    --num-layers "$NUM_LAYERS"
    --hidden-size 4096
    --ffn-hidden-size 14336
    --num-attention-heads 32
    --max-position-embeddings 131072
    --seq-length 8192
    --micro-batch-size 1
    --bf16
)

PREFILL_GPUS=$(seq -s, 0 $((WORLD_PREFILL-1)))
DECODE_GPUS=$(seq -s, $WORLD_PREFILL $((WORLD_PREFILL+WORLD_DECODE-1)))

log "starting owned Megatron PREFILL engine (TP=$TP_PREFILL PP=$PP_PREFILL, GPUs=$PREFILL_GPUS)..."
(
    CUDA_VISIBLE_DEVICES="$PREFILL_GPUS" exec python -m megatron.inference.integrations.dynamo \
            --role prefill \
            --model "$TOKENIZER_MODEL" \
            --served-model-name "$SERVED_MODEL_NAME" \
            --nproc-per-node "$WORLD_PREFILL" \
            --coordinator-host 127.0.0.1 \
            --coordinator-port "$COORD_PORT_PREFILL" \
            --kv-transfer-listen-addr "127.0.0.1:$NIXL_PORT_PREFILL" \
            --megatron-root /opt/megatron-lm \
            -- \
            --tensor-model-parallel-size "$TP_PREFILL" \
            --pipeline-model-parallel-size "$PP_PREFILL" \
            --inference-dynamic-batching-prefix-caching \
            --load "$MODEL_CHECKPOINT" \
            --tokenizer-type HuggingFaceTokenizer \
            --tokenizer-model "$TOKENIZER_MODEL" \
            "${MODEL_ARGS[@]}"
) > "$LOG_DIR/worker-prefill.log" 2>&1 &
PIDS+=($!)

log "starting owned Megatron DECODE engine (TP=$TP_DECODE PP=$PP_DECODE, GPUs=$DECODE_GPUS)..."
(
    CUDA_VISIBLE_DEVICES="$DECODE_GPUS" exec python -m megatron.inference.integrations.dynamo \
            --role decode \
            --model "$TOKENIZER_MODEL" \
            --served-model-name "$SERVED_MODEL_NAME" \
            --nproc-per-node "$WORLD_DECODE" \
            --coordinator-host 127.0.0.1 \
            --coordinator-port "$COORD_PORT_DECODE" \
            --kv-transfer-listen-addr "127.0.0.1:$NIXL_PORT_DECODE" \
            --megatron-root /opt/megatron-lm \
            -- \
            --tensor-model-parallel-size "$TP_DECODE" \
            --pipeline-model-parallel-size "$PP_DECODE" \
            --inference-dynamic-batching-prefix-caching \
            --load "$MODEL_CHECKPOINT" \
            --tokenizer-type HuggingFaceTokenizer \
            --tokenizer-model "$TOKENIZER_MODEL" \
            "${MODEL_ARGS[@]}"
) > "$LOG_DIR/worker-decode.log" 2>&1 &
PIDS+=($!)

wait_for "prefill worker registered" 600 \
    grep -q "Registered base model" "$LOG_DIR/worker-prefill.log" \
    || die "prefill worker never registered (see $LOG_DIR/worker-prefill.log)"
wait_for "decode worker registered" 600 \
    grep -q "Registered base model" "$LOG_DIR/worker-decode.log" \
    || die "decode worker never registered (see $LOG_DIR/worker-decode.log)"

###############################################################################
# 4. Dynamo frontend
###############################################################################
log "starting Dynamo frontend on :$HTTP_PORT..."
python -m dynamo.frontend --http-port "$HTTP_PORT" --router-mode kv \
    --request-plane nats --event-plane nats > "$LOG_DIR/frontend.log" 2>&1 &
PIDS+=($!)
wait_for "frontend exposes $SERVED_MODEL_NAME" 60 \
    bash -c "curl -sf http://127.0.0.1:$HTTP_PORT/v1/models | grep -q '$SERVED_MODEL_NAME'" \
    || die "frontend never exposed model (see $LOG_DIR/frontend.log)"

###############################################################################
# 5. Publish env + block
###############################################################################
cat > /tmp/phase3_hetero.env <<ENV
export PHASE3_FRONTEND_URL="http://127.0.0.1:$HTTP_PORT"
export PHASE3_MODEL_NAME="$SERVED_MODEL_NAME"
export PHASE3_LOG_DIR="$LOG_DIR"
export PHASE3_PREFILL_LOG="$LOG_DIR/worker-prefill.log"
export PHASE3_DECODE_LOG="$LOG_DIR/worker-decode.log"
export TP_PREFILL="$TP_PREFILL"
export TP_DECODE="$TP_DECODE"
export PP_PREFILL="$PP_PREFILL"
export PP_DECODE="$PP_DECODE"
ENV

log "all components healthy. test endpoint: http://127.0.0.1:$HTTP_PORT"
echo "PHASE3_HETERO_READY"
wait -n "${PIDS[@]}"
log "a component exited; tearing down"
exit 1
