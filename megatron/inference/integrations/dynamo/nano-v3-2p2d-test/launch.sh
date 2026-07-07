#!/usr/bin/env bash
# Launch the eight-GPU Nano v3 2P2D test in an existing Slurm allocation.

set -euo pipefail

: "${SLURM_JOB_ID:?must run inside an existing Slurm allocation}"
: "${DMG_SQSH:?DMG_SQSH must point to the Dynamo-Megatron squashfs image}"
: "${STAGE:?STAGE must be a shared directory visible on every node}"

[[ -f "$DMG_SQSH" ]] || { echo "DMG_SQSH not found: $DMG_SQSH" >&2; exit 1; }
[[ -d "$STAGE" ]] || { echo "STAGE not found: $STAGE" >&2; exit 1; }

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MEGATRON_ROOT="${MEGATRON_LOCAL_DEV:-$(cd "$SCRIPT_DIR/../../../../.." && pwd)}"
[[ -d "$MEGATRON_ROOT" ]] || { echo "Megatron root not found: $MEGATRON_ROOT" >&2; exit 1; }

MODEL_CHECKPOINT="${MODEL_CHECKPOINT:-/lustre/fsw/portfolios/llmservice/users/ksanthanam/nemotron-3-nano-30b}"
PRETRAINED_CHECKPOINT="${PRETRAINED_CHECKPOINT:-/lustre/fsw/portfolios/llmservice/users/ksanthanam/nanov3}"
TOKENIZER_MODEL="${TOKENIZER_MODEL:-/lustre/fsw/portfolios/llmservice/projects/llmservice_nlp_fm/nemotron6/tokenizers/multiMixV8.gpt4o_nc_sd.500000.128k.vocab.json}"
DYNAMO_MODEL="${DYNAMO_MODEL:-nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16}"

for artifact in "$MODEL_CHECKPOINT" "$PRETRAINED_CHECKPOINT" "$TOKENIZER_MODEL"; do
    [[ -e "$artifact" ]] || { echo "required artifact not found: $artifact" >&2; exit 1; }
done
if [[ -e "$DYNAMO_MODEL" && ! -d "$DYNAMO_MODEL" ]]; then
    echo "DYNAMO_MODEL must be a directory or Hugging Face model ID, got file: $DYNAMO_MODEL" >&2
    exit 1
fi

ROLE_EP_SIZE="${ROLE_EP_SIZE:-2}"
TEST_NNODES="${TEST_NNODES:-2}"
GPUS_PER_NODE="${GPUS_PER_NODE:-4}"
PREFILL_WORKERS=2
DECODE_WORKERS=2
TOTAL_GPUS=$((TEST_NNODES * GPUS_PER_NODE))
TOTAL_WORKERS=$((PREFILL_WORKERS + DECODE_WORKERS))

if (( ROLE_EP_SIZE != 2 )); then
    echo "the 2P2D test requires ROLE_EP_SIZE=2, got $ROLE_EP_SIZE" >&2
    exit 2
fi
if (( TEST_NNODES != 1 && TEST_NNODES != 2 )); then
    echo "TEST_NNODES must be 1 or 2, got $TEST_NNODES" >&2
    exit 2
fi
if (( TOTAL_GPUS != 8 || GPUS_PER_NODE % ROLE_EP_SIZE != 0 )); then
    echo "topology must total eight GPUs divisible into EP=2 workers; got ${TEST_NNODES}x${GPUS_PER_NODE}" >&2
    exit 2
fi
if (( TOTAL_GPUS / ROLE_EP_SIZE != TOTAL_WORKERS )); then
    echo "eight GPUs must produce exactly four EP=2 workers" >&2
    exit 2
fi

ALLOCATED_NODES="${SLURM_NNODES:-$(scontrol show hostnames "$SLURM_JOB_NODELIST" | wc -l)}"
if (( TEST_NNODES > ALLOCATED_NODES )); then
    echo "requested $TEST_NNODES test nodes but allocation has $ALLOCATED_NODES" >&2
    exit 2
fi
mapfile -t TEST_HOSTS < <(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n "$TEST_NNODES")
HEAD_HOST="${TEST_HOSTS[0]}"
TEST_HOSTS_CSV="$(IFS=,; echo "${TEST_HOSTS[*]}")"
TEST_NODELIST="$TEST_HOSTS_CSV"

RUN_ID="${RUN_ID:-$SLURM_JOB_ID}"
RUN_ROOT="${RUN_ROOT:-$STAGE/nano-v3-2p2d/runs}"
RUN_DIR="$RUN_ROOT/$RUN_ID"
if [[ -e "$RUN_DIR/test-complete" ]]; then
    echo "run already completed at $RUN_DIR; set a new RUN_ID to repeat it" >&2
    exit 2
fi
mkdir -p "$RUN_DIR"

declare -A SEEN_MOUNTS
MOUNTS="$STAGE:$STAGE,$MEGATRON_ROOT:/opt/megatron-lm"
add_mount() {
    local path="$1" dir
    [[ -e "$path" ]] || return 0
    if [[ -d "$path" ]]; then dir="$path"; else dir="$(dirname "$path")"; fi
    if [[ -z "${SEEN_MOUNTS[$dir]:-}" ]]; then
        SEEN_MOUNTS[$dir]=1
        MOUNTS="$MOUNTS,$dir:$dir:ro"
    fi
}
add_mount "$MODEL_CHECKPOINT"
add_mount "$PRETRAINED_CHECKPOINT"
add_mount "$TOKENIZER_MODEL"
add_mount "$DYNAMO_MODEL"
if [[ -n "${EXTRA_MOUNTS:-}" ]]; then MOUNTS="$MOUNTS,$EXTRA_MOUNTS"; fi

export RUN_ID RUN_ROOT RUN_DIR HEAD_HOST TEST_HOSTS_CSV TEST_NNODES GPUS_PER_NODE
export PREFILL_WORKERS DECODE_WORKERS ROLE_EP_SIZE STAGE
export MODEL_CHECKPOINT PRETRAINED_CHECKPOINT TOKENIZER_MODEL DYNAMO_MODEL

EXPORT_VARS="RUN_ID,RUN_ROOT,RUN_DIR,HEAD_HOST,TEST_HOSTS_CSV,TEST_NNODES,GPUS_PER_NODE,STAGE"
EXPORT_VARS="$EXPORT_VARS,PREFILL_WORKERS,DECODE_WORKERS,ROLE_EP_SIZE"
EXPORT_VARS="$EXPORT_VARS,MODEL_CHECKPOINT,PRETRAINED_CHECKPOINT,TOKENIZER_MODEL,DYNAMO_MODEL"
EXPORT_VARS="$EXPORT_VARS,HF_HOME,HF_TOKEN,NAMESPACE,SERVED_MODEL_NAME,MODEL_ARGS_OVERRIDE"
EXPORT_VARS="$EXPORT_VARS,CONTEXT_LENGTH,INFER_MAX_SEQ_LEN,INFER_BUFFER_GB,INFER_MAX_TOKENS,INFER_MAX_REQUESTS"
EXPORT_VARS="$EXPORT_VARS,KV_BLOCK_SIZE,MAMBA_GB,PREFIX_CACHE_EVICTION_POLICY,CUDA_GRAPH_IMPL"
EXPORT_VARS="$EXPORT_VARS,CUDA_DEVICE_MAX_CONNECTIONS,MAMBA_DETERMINISTIC,TRITON_CACHE_AUTOTUNING"
EXPORT_VARS="$EXPORT_VARS,HTTP_PORT,NATS_PORT,NATS_MONITOR_PORT,ETCD_PORT,COORD_PORT_BASE"
EXPORT_VARS="$EXPORT_VARS,WORKER_START_TIMEOUT,TEST_TIMEOUT_SECONDS,FRONTEND_SETTLE_SECONDS"
EXPORT_VARS="$EXPORT_VARS,REQUEST_CONCURRENCY,MAX_REQUESTS,MAX_TOKENS,PROMPT_REPEAT,LOG_SETTLE_SECONDS"

echo "[launch] nodes:      $TEST_NNODES ($TEST_HOSTS_CSV)"
echo "[launch] topology:   2P2D, TP=1 PP=1 EP=2, eight GPUs total"
echo "[launch] placement:  $((GPUS_PER_NODE / ROLE_EP_SIZE)) workers/node"
echo "[launch] run dir:    $RUN_DIR"
echo "[launch] container:  $DMG_SQSH"

exec srun \
    --jobid="$SLURM_JOB_ID" --overlap \
    --nodes="$TEST_NNODES" --ntasks="$TEST_NNODES" --ntasks-per-node=1 \
    --nodelist="$TEST_NODELIST" \
    --gpus-per-node="$GPUS_PER_NODE" \
    --container-image="$DMG_SQSH" \
    --container-mounts="$MOUNTS" \
    --container-workdir=/opt/megatron-lm \
    --export="ALL,$EXPORT_VARS" \
    bash /opt/megatron-lm/megatron/inference/integrations/dynamo/nano-v3-2p2d-test/orchestrate-node.sh
