#!/usr/bin/env bash
# Launch one orchestration task per Slurm node inside the Dynamo-Megatron image.

set -euo pipefail

: "${SLURM_JOB_ID:?must run inside an existing Slurm allocation}"
: "${DMG_SQSH:?DMG_SQSH must point to the Dynamo-Megatron squashfs image}"
: "${STAGE:?STAGE must be a shared directory visible on every node}"
: "${TEST_MODE:?use launch-events.sh, launch-routing.sh, or launch-baseline.sh}"

case "$TEST_MODE" in
    events|routing|baseline) ;;
    *) echo "unsupported TEST_MODE: $TEST_MODE" >&2; exit 2 ;;
esac

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
GPUS_PER_NODE="${GPUS_PER_NODE:-4}"
KV_BLOCK_SIZE="${KV_BLOCK_SIZE:-256}"
PREFIX_CACHE="${PREFIX_CACHE:-1}"
if (( ROLE_EP_SIZE != 2 && ROLE_EP_SIZE != 4 )); then
    echo "ROLE_EP_SIZE must be 2 or 4, got $ROLE_EP_SIZE" >&2
    exit 2
fi
if (( GPUS_PER_NODE % ROLE_EP_SIZE != 0 )); then
    echo "GPUS_PER_NODE=$GPUS_PER_NODE must be divisible by ROLE_EP_SIZE=$ROLE_EP_SIZE" >&2
    exit 2
fi
if (( KV_BLOCK_SIZE < 256 || KV_BLOCK_SIZE % 256 != 0 )); then
    echo "KV_BLOCK_SIZE must be a positive multiple of 256, got $KV_BLOCK_SIZE" >&2
    exit 2
fi
if [[ "$PREFIX_CACHE" != "1" ]]; then
    echo "PREFIX_CACHE must be 1 for KV-event and routing tests" >&2
    exit 2
fi

RUN_ID="${RUN_ID:-${SLURM_JOB_ID}-${TEST_MODE}}"
RUN_ROOT="${RUN_ROOT:-$STAGE/nano-v3-kv-routing/runs}"
RUN_DIR="$RUN_ROOT/$RUN_ID"
DATASET_PATH="${DATASET_PATH:-$STAGE/nano-v3-kv-routing/dataset.jsonl}"
if [[ -e "$RUN_DIR/test-complete" ]]; then
    echo "run already completed at $RUN_DIR; set a new RUN_ID to repeat it" >&2
    exit 2
fi
mkdir -p "$RUN_DIR" "$(dirname "$DATASET_PATH")"

ALLOCATED_NODES="${SLURM_NNODES:-$(scontrol show hostnames "$SLURM_JOB_NODELIST" | wc -l)}"
NNODES="${TEST_NNODES:-2}"
if (( NNODES < 1 || NNODES > ALLOCATED_NODES )); then
    echo "TEST_NNODES must be between 1 and the $ALLOCATED_NODES allocated nodes, got $NNODES" >&2
    exit 2
fi
mapfile -t TEST_HOSTS < <(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n "$NNODES")
HEAD_HOST="${TEST_HOSTS[0]}"
TEST_NODELIST="$(IFS=,; echo "${TEST_HOSTS[*]}")"

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
if [[ -n "${EXTRA_MOUNTS:-}" ]]; then
    MOUNTS="$MOUNTS,$EXTRA_MOUNTS"
fi

export TEST_MODE RUN_ID RUN_ROOT RUN_DIR DATASET_PATH HEAD_HOST NNODES
export MODEL_CHECKPOINT PRETRAINED_CHECKPOINT TOKENIZER_MODEL DYNAMO_MODEL
export ROLE_EP_SIZE GPUS_PER_NODE KV_BLOCK_SIZE PREFIX_CACHE STAGE

EXPORT_VARS="TEST_MODE,RUN_ID,RUN_ROOT,RUN_DIR,DATASET_PATH,HEAD_HOST,NNODES,STAGE"
EXPORT_VARS="$EXPORT_VARS,MODEL_CHECKPOINT,PRETRAINED_CHECKPOINT,TOKENIZER_MODEL,DYNAMO_MODEL"
EXPORT_VARS="$EXPORT_VARS,ROLE_EP_SIZE,GPUS_PER_NODE,HF_HOME,HF_TOKEN,NAMESPACE,SERVED_MODEL_NAME"
EXPORT_VARS="$EXPORT_VARS,CONTEXT_LENGTH,INFER_MAX_SEQ_LEN,INFER_BUFFER_GB,INFER_MAX_TOKENS,INFER_MAX_REQUESTS"
EXPORT_VARS="$EXPORT_VARS,KV_BLOCK_SIZE,MAMBA_GB,PREFIX_CACHE,CUDA_GRAPH_IMPL,MODEL_ARGS_OVERRIDE"
EXPORT_VARS="$EXPORT_VARS,CUDA_DEVICE_MAX_CONNECTIONS,MAMBA_DETERMINISTIC,TRITON_CACHE_AUTOTUNING"
EXPORT_VARS="$EXPORT_VARS,HTTP_PORT,NATS_PORT,NATS_MONITOR_PORT,ETCD_PORT,COORD_PORT_BASE"
EXPORT_VARS="$EXPORT_VARS,WORKER_START_TIMEOUT,TEST_TIMEOUT_SECONDS,FRONTEND_SETTLE_SECONDS"
EXPORT_VARS="$EXPORT_VARS,EVENT_SETTLE_SECONDS,TURN_SETTLE_SECONDS,LOG_SETTLE_SECONDS"
EXPORT_VARS="$EXPORT_VARS,PREFIX_FAMILIES,TURNS_PER_FAMILY,PREFIX_REPEAT,MAX_TOKENS"
EXPORT_VARS="$EXPORT_VARS,ROUTING_CONCURRENCY,BASELINE_CONCURRENCY,MIN_AFFINITY,WORKLOAD_SEED"

echo "[launch] mode:       $TEST_MODE"
echo "[launch] nodes:      $NNODES ($HEAD_HOST is controller)"
echo "[launch] topology:   TP=1 PP=1 EP=$ROLE_EP_SIZE, $((GPUS_PER_NODE / ROLE_EP_SIZE)) workers/node"
echo "[launch] run dir:    $RUN_DIR"
echo "[launch] dataset:    $DATASET_PATH"
echo "[launch] container:  $DMG_SQSH"

exec srun \
    --jobid="$SLURM_JOB_ID" \
    --overlap \
    --nodes="$NNODES" \
    --ntasks="$NNODES" \
    --ntasks-per-node=1 \
    --nodelist="$TEST_NODELIST" \
    --container-image="$DMG_SQSH" \
    --container-mounts="$MOUNTS" \
    --container-workdir=/opt/megatron-lm \
    --export="ALL,$EXPORT_VARS" \
    bash /opt/megatron-lm/megatron/inference/integrations/dynamo/nano-v3-kv-routing-test/orchestrate-node.sh
