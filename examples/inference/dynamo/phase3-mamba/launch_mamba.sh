#!/usr/bin/env bash
# Hybrid-Mamba disagg host wrapper (Nemotron-3 Nano). Run from inside an
# salloc'd shell on the login node. srun's into the dynamo-megatron container
# (enroot/pyxis) and invokes orchestrate.sh, which brings up the prefill+decode
# disagg topology (+ optional aggregated baseline) and transfers Mamba conv/ssm
# state across the handoff.
#
# Required env:
#   SLURM_JOB_ID   set automatically by salloc
#   DMG_SQSH       absolute path to the dynamo-megatron container sqsh. Must have
#                  `nixl` importable: python -c "from nixl._api import nixl_agent"
#   STAGE          lustre staging dir for hf-cache + component logs
#
# Checkpoint / tokenizer (default to the cluster-staged Nano v3 artifacts). The
# wrapper binds each of these into the container at the SAME absolute path so
# orchestrate.sh's --load / --pretrained-checkpoint resolve unchanged:
#   MODEL_CHECKPOINT       (default /lustre/.../ksanthanam/nemotron-3-nano-30b)
#   PRETRAINED_CHECKPOINT  (default /lustre/.../ksanthanam/nanov3)
#   TOKENIZER_MODEL        (Megatron tiktoken vocab restored by the checkpoint)
#   DYNAMO_MODEL           (default nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16)
#
# Optional env (forwarded to orchestrate.sh):
#   SERVED_MODEL_NAME, PREFLIGHT_ONLY, CONTEXT_LENGTH, INFER_BUFFER_GB, INFER_MAX_TOKENS
#   INFER_MAX_REQUESTS, MAMBA_GB, PREFIX_CACHE, ROLE_EP_SIZE
#   WITH_BASELINE, GPU_PREFILL, GPU_DECODE, GPU_BASELINE, MODEL_ARGS_OVERRIDE
#   HTTP_PORT, HTTP_PORT_AGG, COORD_PORT_*, NIXL_PORT_*, MASTER_PORT_*
#   EXTRA_MOUNTS  extra comma-separated src:dst binds (e.g. your own ckpt dir)
#   MEGATRON_LOCAL_DEV          host dir -> /opt/megatron-lm (live core edits)
#
# Emits "PHASE3_MAMBA_READY" on stdout when all components are healthy.

set -euo pipefail

: "${SLURM_JOB_ID:?must run inside a salloc allocation}"
: "${DMG_SQSH:?DMG_SQSH must be set (path to the dynamo-megatron sqsh on lustre)}"
: "${STAGE:?STAGE must be set (lustre staging dir)}"

[[ -f "$DMG_SQSH" ]] || { echo "DMG_SQSH not found: $DMG_SQSH" >&2; exit 1; }
[[ -d "$STAGE"   ]] || { echo "STAGE not found: $STAGE"        >&2; exit 1; }

# Defaults must match orchestrate.sh so the mounts cover what it will --load.
MODEL_CHECKPOINT="${MODEL_CHECKPOINT:-/lustre/fsw/portfolios/llmservice/users/ksanthanam/nemotron-3-nano-30b}"
PRETRAINED_CHECKPOINT="${PRETRAINED_CHECKPOINT:-/lustre/fsw/portfolios/llmservice/users/ksanthanam/nanov3}"
TOKENIZER_MODEL="${TOKENIZER_MODEL:-/lustre/fsw/portfolios/llmservice/projects/llmservice_nlp_fm/nemotron6/tokenizers/multiMixV8.gpt4o_nc_sd.500000.128k.vocab.json}"
DYNAMO_MODEL="${DYNAMO_MODEL:-nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16}"
export MODEL_CHECKPOINT PRETRAINED_CHECKPOINT TOKENIZER_MODEL DYNAMO_MODEL

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MEGATRON_ROOT="${MEGATRON_LOCAL_DEV:-$(cd "$SCRIPT_DIR/../../../.." && pwd)}"
[[ -d "$MEGATRON_ROOT" ]] || { echo "MEGATRON_ROOT not a dir: $MEGATRON_ROOT" >&2; exit 1; }

# Build the mount list. Bind each artifact's directory at the same path inside
# the container (read-only — checkpoints are never written). A directory is
# added once even if several artifacts share it (e.g. ckpt + pretrained under
# the same users/ dir).
declare -A SEEN
MOUNTS="$STAGE:$STAGE,$MEGATRON_ROOT:/opt/megatron-lm"

add_mount() {
    local host_path="$1"
    [[ -e "$host_path" ]] || { echo "checkpoint path not found: $host_path" >&2; exit 1; }
    # A file (the tokenizer vocab) is reached by binding its parent directory.
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

# Forward env that orchestrate.sh consumes.
EXPORT_VARS="STAGE,HF_HOME,HF_TOKEN,MODEL_CHECKPOINT,PRETRAINED_CHECKPOINT,TOKENIZER_MODEL,DYNAMO_MODEL"
EXPORT_VARS="$EXPORT_VARS,SERVED_MODEL_NAME,PREFLIGHT_ONLY,CONTEXT_LENGTH,INFER_MAX_SEQ_LEN,INFER_BUFFER_GB,INFER_MAX_TOKENS,INFER_MAX_REQUESTS"
EXPORT_VARS="$EXPORT_VARS,MAMBA_GB,PREFIX_CACHE,ROLE_EP_SIZE"
EXPORT_VARS="$EXPORT_VARS,WITH_BASELINE,GPU_PREFILL,GPU_DECODE,GPU_BASELINE,MODEL_ARGS_OVERRIDE"
EXPORT_VARS="$EXPORT_VARS,HTTP_PORT,HTTP_PORT_AGG,COORD_PORT_PREFILL,COORD_PORT_DECODE,COORD_PORT_AGG"
EXPORT_VARS="$EXPORT_VARS,NIXL_PORT_PREFILL,NIXL_PORT_DECODE"
EXPORT_VARS="$EXPORT_VARS,MASTER_PORT_PREFILL,MASTER_PORT_DECODE,MASTER_PORT_AGG"

# UCX transport constraints (must override the image's UCX_TLS=tcp; see
# launch_hetero_tp.sh for why TCP is fatal for VRAM transfers on aarch64).
UCX_TLS="${UCX_TLS_OVERRIDE:-cuda_ipc,cuda_copy,tcp,shm,cma,self}"
UCX_MEMTYPE_CACHE="${UCX_MEMTYPE_CACHE_OVERRIDE:-n}"
UCX_LOG_LEVEL="${UCX_LOG_LEVEL_OVERRIDE:-info}"
UCX_LOG_FILE="${UCX_LOG_FILE_OVERRIDE:-/tmp/ucx_%p.log}"
export UCX_TLS UCX_MEMTYPE_CACHE UCX_LOG_LEVEL UCX_LOG_FILE

echo "[launch] container: $DMG_SQSH"
echo "[launch] mounts:    $MOUNTS"
echo "[launch] expect 'PHASE3_MAMBA_READY' on stdout when ready"
echo

exec srun \
    --jobid="$SLURM_JOB_ID" --overlap \
    --container-image="$DMG_SQSH" \
    --container-name=dmg \
    --container-mounts="$MOUNTS" \
    --container-workdir=/opt/megatron-lm \
    --export="ALL,$EXPORT_VARS,UCX_TLS=$UCX_TLS,UCX_MEMTYPE_CACHE=$UCX_MEMTYPE_CACHE,UCX_LOG_LEVEL=$UCX_LOG_LEVEL,UCX_LOG_FILE=$UCX_LOG_FILE" \
    bash /opt/megatron-lm/examples/inference/dynamo/phase3-mamba/orchestrate.sh
