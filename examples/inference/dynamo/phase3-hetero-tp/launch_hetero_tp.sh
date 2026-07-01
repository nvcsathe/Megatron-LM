#!/usr/bin/env bash
# Heterogeneous-TP host wrapper. Run from inside an existing salloc'd shell on
# the login node. srun's into the dynamo-megatron container (enroot/pyxis) and
# invokes orchestrate.sh, which brings up the two-engine (prefill + decode)
# disagg topology with different TP degrees on each side.
#
# Required env:
#   SLURM_JOB_ID   set automatically by salloc
#   DMG_SQSH       absolute path to the dynamo-megatron container sqsh.
#                  The container must have `nixl` Python + native runtime
#                  importable — confirm with:
#                      python -c "from nixl._api import nixl_agent"
#                  inside the container before relying on this launcher.
#   STAGE          lustre staging dir holding the model checkpoint + hf-cache
#
# Optional env (passed through to orchestrate.sh):
#   MODEL_CHECKPOINT, TOKENIZER_MODEL, SERVED_MODEL_NAME, CONTEXT_LENGTH
#   TP_PREFILL, TP_DECODE         (default 2, 4 — must both divide NUM_QUERY_GROUPS)
#   NUM_QUERY_GROUPS              (default 8 for Llama-3.1-8B)
#   HTTP_PORT
#   COORD_PORT_PREFILL, COORD_PORT_DECODE     (default 5555, 5556)
#   NIXL_PORT_PREFILL, NIXL_PORT_DECODE       (default 7000, 7001)
#   MASTER_PORT_PREFILL, MASTER_PORT_DECODE   (default 29500, 29501)
#   MEGATRON_LOCAL_DEV
#
# UCX transport / logging (force-set in this script, use _OVERRIDE suffix to change):
#   UCX_TLS_OVERRIDE            transport allow-list (default: cuda_ipc,cuda_copy,tcp,shm,cma,self)
#                               TCP is intentionally excluded from pure-VRAM paths; see orchestrate.sh.
#   UCX_MEMTYPE_CACHE_OVERRIDE  set to "n" by default to avoid early-init misclassification
#   UCX_LOG_LEVEL_OVERRIDE      UCX log verbosity (default: info — set to warn once stable)
#   UCX_LOG_FILE_OVERRIDE       log path; UCX expands %p to PID (default: /tmp/ucx_%p.log)
#
# If MEGATRON_LOCAL_DEV is set, the host directory it points to is mounted
# over /opt/megatron-lm so you can edit InferenceClient, coordinator, and
# disaggregation package code without rebuilding the image.
#
# Emits "PHASE3_HETERO_READY" on stdout when all components are healthy.

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

# Forward env that orchestrate.sh consumes. HF_TOKEN is needed for gated
# repos like meta-llama/Llama-3.1-8B.
EXPORT_VARS="STAGE,HF_HOME,HF_TOKEN,MODEL_CHECKPOINT,MODEL_DIR,TOKENIZER_MODEL,SERVED_MODEL_NAME"
EXPORT_VARS="$EXPORT_VARS,CONTEXT_LENGTH,TP_PREFILL,TP_DECODE,NUM_QUERY_GROUPS,HTTP_PORT"
EXPORT_VARS="$EXPORT_VARS,COORD_PORT_PREFILL,COORD_PORT_DECODE"
EXPORT_VARS="$EXPORT_VARS,NIXL_PORT_PREFILL,NIXL_PORT_DECODE"
EXPORT_VARS="$EXPORT_VARS,MASTER_PORT_PREFILL,MASTER_PORT_DECODE"

# UCX transport constraints and diagnostic logging.
#
# The NGC base image bakes in UCX_TLS=tcp to keep NCCL off UCX-RDMA paths.
# That setting is fatal for NIXL: uct_tcp_ep_am_bcopy on the prefill side
# tries to CPU-memcpy from VRAM addresses when handling decode's GET request
# → SIGSEGV on aarch64. We must force-override it unconditionally (no :=
# setdefault — that silently skips if the var is already set to "tcp").
#
# Passing the override both as an exported shell var and via srun --env
# ensures pyxis applies it on top of the image ENV layer.
UCX_TLS="${UCX_TLS_OVERRIDE:-cuda_ipc,cuda_copy,tcp,shm,cma,self}"
UCX_MEMTYPE_CACHE="${UCX_MEMTYPE_CACHE_OVERRIDE:-n}"
UCX_LOG_LEVEL="${UCX_LOG_LEVEL_OVERRIDE:-info}"
UCX_LOG_FILE="${UCX_LOG_FILE_OVERRIDE:-/tmp/ucx_%p.log}"
export UCX_TLS UCX_MEMTYPE_CACHE UCX_LOG_LEVEL UCX_LOG_FILE

echo "[launch] container: $DMG_SQSH"
echo "[launch] mounts:    $MOUNTS"
echo "[launch] UCX_TLS=$UCX_TLS  UCX_LOG_FILE=$UCX_LOG_FILE"
echo "[launch] TP_PREFILL=${TP_PREFILL:-2}  TP_DECODE=${TP_DECODE:-4}  NUM_QUERY_GROUPS=${NUM_QUERY_GROUPS:-8}"
echo "[launch] expect 'PHASE3_HETERO_READY' on stdout when ready"
echo

exec srun \
    --jobid="$SLURM_JOB_ID" --overlap \
    --container-image="$DMG_SQSH" \
    --container-name=dmg \
    --container-mounts="$MOUNTS" \
    --container-workdir=/opt/megatron-lm \
    --export="ALL,$EXPORT_VARS,UCX_TLS=$UCX_TLS,UCX_MEMTYPE_CACHE=$UCX_MEMTYPE_CACHE,UCX_LOG_LEVEL=$UCX_LOG_LEVEL,UCX_LOG_FILE=$UCX_LOG_FILE" \
    bash /opt/megatron-lm/examples/inference/dynamo/phase3-hetero-tp/orchestrate.sh
