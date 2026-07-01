#!/usr/bin/env bash
# Phase-0 host wrapper. Run from inside an existing salloc'd shell on the
# login node. srun's into the dynamo-megatron container (enroot/pyxis) and
# invokes orchestrate.sh.
#
# Required env:
#   SLURM_JOB_ID   set automatically by salloc
#   DMG_SQSH       absolute path to chaitrasathe+dynamo-megatron+phase0-arm64.sqsh
#   STAGE          lustre staging dir holding the model checkpoint + hf-cache
#
# Optional env (passed through to orchestrate.sh):
#   MODEL_CHECKPOINT, TOKENIZER_MODEL, SERVED_MODEL_NAME, TP, HTTP_PORT
#   COORD_PORT, CONTEXT_LENGTH, MASTER_PORT, MEGATRON_LOCAL_DEV
#
# If MEGATRON_LOCAL_DEV is set, the host directory it points to is mounted
# over /opt/megatron-lm so you can edit InferenceClient / coordinator code
# without rebuilding the image.

set -euo pipefail

: "${SLURM_JOB_ID:?must run inside a salloc allocation}"
: "${DMG_SQSH:?DMG_SQSH must be set (path to the phase0 sqsh on lustre)}"
: "${STAGE:?STAGE must be set (lustre staging dir)}"

[[ -f "$DMG_SQSH" ]] || { echo "DMG_SQSH not found: $DMG_SQSH" >&2; exit 1; }
[[ -d "$STAGE"   ]] || { echo "STAGE not found: $STAGE"        >&2; exit 1; }

# Mount this Megatron checkout so engine, adapter, and launcher edits are all
# visible without rebuilding the development image.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MEGATRON_ROOT="${MEGATRON_LOCAL_DEV:-$(cd "$SCRIPT_DIR/../../../.." && pwd)}"
[[ -d "$MEGATRON_ROOT" ]] || { echo "MEGATRON_ROOT not a dir: $MEGATRON_ROOT" >&2; exit 1; }
MOUNTS="$STAGE:$STAGE,$MEGATRON_ROOT:/opt/megatron-lm"
echo "[launch] Megatron source mount: $MEGATRON_ROOT -> /opt/megatron-lm"

# Forward env that orchestrate.sh consumes. HF_TOKEN is needed for gated
# repos like meta-llama/Llama-3.1-8B.
EXPORT_VARS="STAGE,HF_HOME,HF_TOKEN,MODEL_CHECKPOINT,MODEL_DIR,TOKENIZER_MODEL,SERVED_MODEL_NAME"
EXPORT_VARS="$EXPORT_VARS,CONTEXT_LENGTH,TP,HTTP_PORT,COORD_PORT,MASTER_PORT"

echo "[launch] container: $DMG_SQSH"
echo "[launch] mounts:    $MOUNTS"
echo "[launch] expect 'PHASE0_READY' on stdout when ready"
echo

exec srun \
    --jobid="$SLURM_JOB_ID" --overlap \
    --container-image="$DMG_SQSH" \
    --container-name=dmg \
    --container-mounts="$MOUNTS" \
    --container-workdir=/opt/megatron-lm \
    --export="ALL,$EXPORT_VARS" \
    bash /opt/megatron-lm/examples/inference/dynamo/phase0/orchestrate.sh
