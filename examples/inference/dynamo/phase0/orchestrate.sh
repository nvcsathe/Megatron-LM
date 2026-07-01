#!/usr/bin/env bash
# Phase-0 orchestrator. Runs INSIDE the dynamo-megatron container.
# Brings up NATS, etcd, one self-owned Dynamo Megatron engine (TP=2),
# and the Dynamo frontend — in order, waiting for each
# to be healthy before launching the next. Blocks until Ctrl-C or one
# of the long-running processes dies.
#
# Outputs:
#   - /tmp/phase0.env       : env-vars sourced by test scripts
#   - /tmp/{nats,etcd,worker,frontend}.log
#   - On stdout: a single "PHASE0_READY" line once the frontend serves
#     /v1/models with the target model registered.

set -uo pipefail

STAGE="${STAGE:-/lustre/fsw/portfolios/nemotron/users/csathe}"
MODEL_DIR="${MODEL_DIR:-llama3.1-8b-instruct-mcore}"
MODEL_CHECKPOINT="${MODEL_CHECKPOINT:-$STAGE/models/${MODEL_DIR}}"
TOKENIZER_MODEL="${TOKENIZER_MODEL:-meta-llama/Llama-3.1-8B-Instruct}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-llama-3.1-8b-instruct}"
CONTEXT_LENGTH="${CONTEXT_LENGTH:-4096}"
TP="${TP:-2}"
HTTP_PORT="${HTTP_PORT:-8100}"
COORD_PORT="${COORD_PORT:-5555}"

export NATS_SERVER="nats://127.0.0.1:4222"
export ETCD_ENDPOINTS="http://127.0.0.1:2379"
export HF_HOME="${HF_HOME:-${STAGE}/hf-cache}"
export CUDA_DEVICE_MAX_CONNECTIONS=1
export MASTER_ADDR=127.0.0.1
export MASTER_PORT="${MASTER_PORT:-29500}"

LOG_DIR="${LOG_DIR:-/tmp}"
PIDS=()

log()  { printf '[orchestrate %s] %s\n' "$(date +%H:%M:%S)" "$*"; }
die()  { log "FATAL: $*" >&2; cleanup; exit 1; }

cleanup() {
    log "cleaning up..."
    for pid in "${PIDS[@]}"; do
        kill "$pid" 2>/dev/null || true
    done
    wait 2>/dev/null || true
}
trap cleanup EXIT INT TERM

wait_for() {
    # wait_for <desc> <max-seconds> <command...>
    local desc="$1" max="$2"; shift 2
    local elapsed=0
    while ! "$@" >/dev/null 2>&1; do
        sleep 2
        elapsed=$((elapsed + 2))
        if [[ $elapsed -ge $max ]]; then
            log "TIMEOUT waiting for $desc after ${max}s"
            return 1
        fi
    done
    log "ready: $desc (${elapsed}s)"
}

###############################################################################
# 1. NATS + etcd
###############################################################################
log "starting NATS..."
# -m 8222 exposes the HTTP monitoring endpoint we curl for /healthz below.
# Without it nats-server runs fine on :4222 but the healthcheck times out.
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
# 2. Self-owned Dynamo Megatron engine (TP=$TP)
###############################################################################
# Model-architecture flags. Lifted verbatim from
# Megatron-LM/examples/inference/llama_mistral/run_text_generation_llama3.1.sh
# — Megatron's arg validator requires these even though the checkpoint
# already carries the geometry. Override for non-Llama-3.1-8B models.
MODEL_ARGS=(
    --ckpt-format torch_dist
    --use-checkpoint-args
    --disable-bias-linear
    --transformer-impl transformer_engine
    --normalization RMSNorm
    --group-query-attention --num-query-groups 8
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
    --num-layers 32
    --hidden-size 4096
    --ffn-hidden-size 14336
    --num-attention-heads 32
    --max-position-embeddings 131072
    --seq-length 8192
    --micro-batch-size 1
    --bf16
)

log "starting owned Dynamo Megatron engine (TP=$TP, model=$MODEL_CHECKPOINT)..."
(
    exec python -m megatron.inference.integrations.dynamo \
            --role aggregated \
            --model "$TOKENIZER_MODEL" \
            --served-model-name "$SERVED_MODEL_NAME" \
            --nproc-per-node "$TP" \
            --coordinator-host 127.0.0.1 \
            --coordinator-port "$COORD_PORT" \
            --megatron-root /opt/megatron-lm \
            -- \
            --tensor-model-parallel-size "$TP" \
            --pipeline-model-parallel-size 1 \
            --inference-dynamic-batching-prefix-caching \
            --load "$MODEL_CHECKPOINT" \
            --tokenizer-type HuggingFaceTokenizer \
            --tokenizer-model "$TOKENIZER_MODEL" \
            "${MODEL_ARGS[@]}"
) > "$LOG_DIR/worker.log" 2>&1 &
PIDS+=($!)

wait_for "worker registered" 600 \
    grep -q "Registered base model" "$LOG_DIR/worker.log" \
    || die "worker never registered (see $LOG_DIR/worker.log)"

###############################################################################
# 4. Dynamo frontend (HTTP)
###############################################################################
log "starting Dynamo frontend on :$HTTP_PORT..."
# Pin both planes to NATS — the worker registers on NATS via
# create_runtime(request_plane="nats", event_plane="nats"). On some image
# builds the frontend defaults to TCP, which then fails to dial the
# worker's NATS subject as a socket address.
python -m dynamo.frontend \
    --http-port "$HTTP_PORT" \
    --router-mode kv \
    --request-plane nats \
    --event-plane nats \
    > "$LOG_DIR/frontend.log" 2>&1 &
PIDS+=($!)

# Frontend is ready when /v1/models returns the served model name.
wait_for "frontend exposes $SERVED_MODEL_NAME" 60 \
    bash -c "curl -sf http://127.0.0.1:$HTTP_PORT/v1/models | grep -q '$SERVED_MODEL_NAME'" \
    || die "frontend never exposed model (see $LOG_DIR/frontend.log)"

###############################################################################
# 5. Publish env + block
###############################################################################
cat > /tmp/phase0.env <<ENV
export NATS_SERVER="$NATS_SERVER"
export ETCD_ENDPOINTS="$ETCD_ENDPOINTS"
export MEGATRON_COORDINATOR_ADDR="tcp://127.0.0.1:$COORD_PORT"
export PHASE0_FRONTEND_URL="http://127.0.0.1:$HTTP_PORT"
export PHASE0_MODEL_NAME="$SERVED_MODEL_NAME"
ENV

log "all components healthy. test endpoint: http://127.0.0.1:$HTTP_PORT"
echo "PHASE0_READY"

# Block on whichever long-running process exits first.
wait -n "${PIDS[@]}"
log "a component exited; tearing down"
exit 1
