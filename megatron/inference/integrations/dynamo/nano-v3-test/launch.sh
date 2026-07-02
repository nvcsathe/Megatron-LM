#!/usr/bin/env bash
# Launch the Nano v3 Mamba disaggregation test in an existing Slurm allocation.
# Required: SLURM_JOB_ID, DMG_SQSH, and STAGE. See README.md for overrides.

set -euo pipefail

: "${SLURM_JOB_ID:?must run inside a salloc allocation}"
: "${DMG_SQSH:?DMG_SQSH must be set (path to the dynamo-megatron sqsh on lustre)}"
: "${STAGE:?STAGE must be set (lustre staging dir)}"

[[ -f "$DMG_SQSH" ]] || { echo "DMG_SQSH not found: $DMG_SQSH" >&2; exit 1; }
[[ -d "$STAGE"   ]] || { echo "STAGE not found: $STAGE"        >&2; exit 1; }

# Keep artifact defaults aligned with orchestrate.sh.
MODEL_CHECKPOINT="${MODEL_CHECKPOINT:-/lustre/fsw/portfolios/llmservice/users/ksanthanam/nemotron-3-nano-30b}"
PRETRAINED_CHECKPOINT="${PRETRAINED_CHECKPOINT:-/lustre/fsw/portfolios/llmservice/users/ksanthanam/nanov3}"
TOKENIZER_MODEL="${TOKENIZER_MODEL:-/lustre/fsw/portfolios/llmservice/projects/llmservice_nlp_fm/nemotron6/tokenizers/multiMixV8.gpt4o_nc_sd.500000.128k.vocab.json}"
DYNAMO_MODEL="${DYNAMO_MODEL:-nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16}"
export MODEL_CHECKPOINT PRETRAINED_CHECKPOINT TOKENIZER_MODEL DYNAMO_MODEL

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MEGATRON_ROOT="${MEGATRON_LOCAL_DEV:-$(cd "$SCRIPT_DIR/../../../../.." && pwd)}"
[[ -d "$MEGATRON_ROOT" ]] || { echo "MEGATRON_ROOT not a dir: $MEGATRON_ROOT" >&2; exit 1; }

# Bind artifact directories read-only at their original paths.
declare -A SEEN
MOUNTS="$STAGE:$STAGE,$MEGATRON_ROOT:/opt/megatron-lm"

add_mount() {
    local host_path="$1"
    [[ -e "$host_path" ]] || { echo "checkpoint path not found: $host_path" >&2; exit 1; }
    local dir
    if [[ -d "$host_path" ]]; then dir="$host_path"; else dir="$(dirname "$host_path")"; fi
    if [[ -z "${SEEN[$dir]:-}" ]]; then
        SEEN[$dir]=1
        MOUNTS="$MOUNTS,$dir:$dir:ro"
        echo "[launch] checkpoint mount (ro): $dir"
    fi
}

add_mount "$MODEL_CHECKPOINT"
add_mount "$PRETRAINED_CHECKPOINT"
add_mount "$TOKENIZER_MODEL"
if [[ -e "$DYNAMO_MODEL" ]]; then
    add_mount "$DYNAMO_MODEL"
fi

if [[ -n "${EXTRA_MOUNTS:-}" ]]; then
    MOUNTS="$MOUNTS,$EXTRA_MOUNTS"
    echo "[launch] extra mounts: $EXTRA_MOUNTS"
fi

EXPORT_VARS="STAGE,HF_HOME,HF_TOKEN,MODEL_CHECKPOINT,PRETRAINED_CHECKPOINT,TOKENIZER_MODEL,DYNAMO_MODEL"
EXPORT_VARS="$EXPORT_VARS,SERVED_MODEL_NAME,PREFLIGHT_ONLY,CONTEXT_LENGTH,INFER_MAX_SEQ_LEN,INFER_BUFFER_GB,INFER_MAX_TOKENS,INFER_MAX_REQUESTS"
EXPORT_VARS="$EXPORT_VARS,MAMBA_GB,PREFIX_CACHE,ROLE_EP_SIZE"
EXPORT_VARS="$EXPORT_VARS,WITH_BASELINE,GPU_PREFILL,GPU_DECODE,GPU_BASELINE,MODEL_ARGS_OVERRIDE"
EXPORT_VARS="$EXPORT_VARS,HTTP_PORT,HTTP_PORT_AGG,COORD_PORT_PREFILL,COORD_PORT_DECODE,COORD_PORT_AGG"
EXPORT_VARS="$EXPORT_VARS,NIXL_PORT_PREFILL,NIXL_PORT_DECODE"
EXPORT_VARS="$EXPORT_VARS,MASTER_PORT_PREFILL,MASTER_PORT_DECODE,MASTER_PORT_AGG"

# Override the image transport settings for NIXL VRAM transfers.
UCX_TLS="${UCX_TLS_OVERRIDE:-cuda_ipc,cuda_copy,tcp,shm,cma,self}"
UCX_MEMTYPE_CACHE="${UCX_MEMTYPE_CACHE_OVERRIDE:-n}"
UCX_LOG_LEVEL="${UCX_LOG_LEVEL_OVERRIDE:-info}"
UCX_LOG_FILE="${UCX_LOG_FILE_OVERRIDE:-/tmp/ucx_%p.log}"
export UCX_TLS UCX_MEMTYPE_CACHE UCX_LOG_LEVEL UCX_LOG_FILE

echo "[launch] container: $DMG_SQSH"
echo "[launch] mounts:    $MOUNTS"
echo "[launch] expect 'NANO_V3_TEST_READY' on stdout when ready"
echo

exec srun \
    --jobid="$SLURM_JOB_ID" --overlap \
    --container-image="$DMG_SQSH" \
    --container-name=dmg \
    --container-mounts="$MOUNTS" \
    --container-workdir=/opt/megatron-lm \
    --export="ALL,$EXPORT_VARS,UCX_TLS=$UCX_TLS,UCX_MEMTYPE_CACHE=$UCX_MEMTYPE_CACHE,UCX_LOG_LEVEL=$UCX_LOG_LEVEL,UCX_LOG_FILE=$UCX_LOG_FILE" \
    bash /opt/megatron-lm/megatron/inference/integrations/dynamo/nano-v3-test/orchestrate.sh
