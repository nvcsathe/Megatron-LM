#!/usr/bin/env bash
# Two-node aggregated Nemotron Ultra BF16 deployment:
#   8x GB200 total, TP=8 / EP=8 / expert-TP=1 / dense-DP=1
#
# Submit with:
#   sbatch --account=<account> --partition=<partition> launch_slurm.sh

#SBATCH --job-name=ultra-agg
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --time=04:00:00
#SBATCH --output=ultra-agg-%j.out

set -euo pipefail

export MEGATRON_ROOT="${MEGATRON_ROOT:-${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR is required}}"
export WORKER_SCRIPT="${MEGATRON_ROOT}/examples/inference/nemotron_ultra_agg/launch_slurm.sh"
if [[ ! -f "${WORKER_SCRIPT}" ]]; then
    echo "Worker script is not visible at ${WORKER_SCRIPT}." >&2
    echo "Submit from the shared Megatron-LM worktree or export MEGATRON_ROOT explicitly." >&2
    exit 2
fi

export PRETRAINED_CHECKPOINT="${PRETRAINED_CHECKPOINT:-/lustre/fsw/portfolios/llmservice/projects/llmservice_nlp_fm/nemotron6/55b_hybrid_moe/checkpoints/pre_training_final_lc}"
export LOAD_CHECKPOINT="${LOAD_CHECKPOINT:-/lustre/fs1/portfolios/llmservice/projects/llmservice_fm_text/users/sasatheesh/data/checkpoints/inescapable-sawfly-step108-mcore}"
export TOKENIZER_MODEL="${TOKENIZER_MODEL:-nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16}"
export CONTAINER_IMAGE="${CONTAINER_IMAGE:-/lustre/fsw/portfolios/nemotron/users/csathe/chaitrasathe+dynamo-megatron+mamba.sqsh}"
export CONTAINER_MOUNTS="${CONTAINER_MOUNTS:-/home:/home,/lustre:/lustre}"
export PYTHON_DEPS_DIR="${PYTHON_DEPS_DIR:-${MEGATRON_ROOT}/.runtime-deps/nemotron-ultra-py312}"
export PYTHONPATH="${PYTHON_DEPS_DIR}:${PYTHONPATH:-}"
export HF_HOME="${HF_HOME:-${MEGATRON_ROOT}/.runtime-deps/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"

if [[ ! -f "${CONTAINER_IMAGE}" ]]; then
    echo "Missing container image: ${CONTAINER_IMAGE}" >&2
    exit 2
fi

export EXPECTED_NNODES=2
export GPUS_PER_NODE="${GPUS_PER_NODE:-4}"
export WORLD_SIZE=$((EXPECTED_NNODES * GPUS_PER_NODE))
export TP_SIZE="${TP_SIZE:-8}"
export EP_SIZE="${EP_SIZE:-8}"
if (( WORLD_SIZE != 8 || TP_SIZE != 8 || EP_SIZE != 8 )); then
    echo "Ultra BF16 requires WORLD_SIZE=8, TP_SIZE=8, and EP_SIZE=8; got ${WORLD_SIZE}, ${TP_SIZE}, and ${EP_SIZE}." >&2
    exit 2
fi

export SERVER_PORT="${SERVER_PORT:-8000}"
export COORDINATOR_PORT="${COORDINATOR_PORT:-5555}"
export MASTER_PORT="${MASTER_PORT:-29500}"
export STARTUP_TIMEOUT_SECONDS="${STARTUP_TIMEOUT_SECONDS:-3600}"

# Match the official 8-GPU Ultra deployment profile where Megatron has an
# equivalent option. The KV buffer is Megatron-specific and remains overrideable.
export INFERENCE_MAX_SEQ_LENGTH="${INFERENCE_MAX_SEQ_LENGTH:-262144}"
export INFERENCE_MAX_TOKENS="${INFERENCE_MAX_TOKENS:-32768}"
export INFERENCE_MAX_REQUESTS="${INFERENCE_MAX_REQUESTS:-16}"
export INFERENCE_BUFFER_SIZE_GB="${INFERENCE_BUFFER_SIZE_GB:-25}"
export NUM_SPECULATIVE_TOKENS="${NUM_SPECULATIVE_TOKENS:-5}"
export ENABLE_PREFIX_CACHING="${ENABLE_PREFIX_CACHING:-0}"
export MAMBA_PREFIX_CACHE_GB="${MAMBA_PREFIX_CACHE_GB:-4}"

# GB200 does not require a CUDA_DEVICE_MAX_CONNECTIONS override.
export UCX_MEMTYPE_CACHE="${UCX_MEMTYPE_CACHE:-n}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

run_worker() {
    local node_rank="${SLURM_NODEID:?SLURM_NODEID is required}"
    local -a prefix_cache_args=()
    if [[ "${ENABLE_PREFIX_CACHING}" == "1" ]]; then
        prefix_cache_args=(
            --inference-dynamic-batching-prefix-caching
            --inference-dynamic-batching-prefix-caching-eviction-policy lru
            --inference-dynamic-batching-prefix-caching-mamba-gb "${MAMBA_PREFIX_CACHE_GB}"
        )
    fi

    cd "${MEGATRON_ROOT}"
    # Leave coordinator-host unset so each node binds its MP ZMQ sockets to
    # its own hostname. The shared rank-0 coordinator address is broadcast.
    exec python -m torch.distributed.run \
        --nnodes="${EXPECTED_NNODES}" \
        --nproc-per-node="${GPUS_PER_NODE}" \
        --node-rank="${node_rank}" \
        --master-addr="${MASTER_ADDR}" \
        --master-port="${MASTER_PORT}" \
        -m examples.inference.launch_inference_server \
        --pretrained-checkpoint "${PRETRAINED_CHECKPOINT}" \
        --load "${LOAD_CHECKPOINT}" \
        --tokenizer-type HuggingFaceTokenizer \
        --tokenizer-model "${TOKENIZER_MODEL}" \
        --no-use-tokenizer-model-from-checkpoint-args \
        --model-provider mamba \
        --tensor-model-parallel-size "${TP_SIZE}" \
        --pipeline-model-parallel-size 1 \
        --expert-model-parallel-size "${EP_SIZE}" \
        --expert-tensor-parallel-size 1 \
        --sequence-parallel \
        --moe-router-score-function sigmoid \
        --moe-router-enable-expert-bias \
        --moe-router-topk-scaling-factor 2.5 \
        --moe-router-dtype fp32 \
        --moe-token-dispatcher-type alltoall \
        --moe-grouped-gemm \
        --moe-shared-expert-overlap \
        --use-checkpoint-args \
        --dist-ckpt-strictness log_unexpected \
        --seq-length "${INFERENCE_MAX_SEQ_LENGTH}" \
        --max-position-embeddings "${INFERENCE_MAX_SEQ_LENGTH}" \
        --inference-max-seq-length "${INFERENCE_MAX_SEQ_LENGTH}" \
        --transformer-impl inference_optimized \
        --inference-grouped-gemm-backend vllm \
        --inference-use-synchronous-zmq-collectives \
        --mtp-use-repeated-layer \
        --num-speculative-tokens "${NUM_SPECULATIVE_TOKENS}" \
        --mamba-inference-ssm-states-dtype fp32 \
        --micro-batch-size 1 \
        --bf16 \
        --no-load-optim \
        --distributed-backend nccl \
        --inference-dynamic-batching \
        --inference-dynamic-batching-buffer-size-gb "${INFERENCE_BUFFER_SIZE_GB}" \
        --inference-dynamic-batching-max-tokens "${INFERENCE_MAX_TOKENS}" \
        --inference-dynamic-batching-max-requests "${INFERENCE_MAX_REQUESTS}" \
        --enable-chunked-prefill \
        "${prefix_cache_args[@]}" \
        --coordinator-port "${COORDINATOR_PORT}" \
        --host 0.0.0.0 \
        --port "${SERVER_PORT}" \
        --frontend-replicas 1 \
        --parsers nemotron-v3-reasoning qwen3-coder-tool
}

if [[ "${1:-}" == "--worker" ]]; then
    run_worker
fi

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    echo "Run this script with sbatch (it requires two allocated nodes)." >&2
    exit 2
fi
if [[ "${SLURM_NNODES}" -ne "${EXPECTED_NNODES}" ]]; then
    echo "Expected ${EXPECTED_NNODES} nodes (8 GB200 GPUs total); got ${SLURM_NNODES}." >&2
    exit 2
fi

export MASTER_ADDR="${MASTER_ADDR:-$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n 1)}"
export ENDPOINT_HOST="${ENDPOINT_HOST:-${MASTER_ADDR}}"
export ENDPOINT_FILE="${ENDPOINT_FILE:-${MEGATRON_ROOT}/ultra-agg-${SLURM_JOB_ID}.endpoint}"

echo "Checking Quart/Hypercorn in ${PYTHON_DEPS_DIR}."
srun \
    --nodes=1 \
    --ntasks=1 \
    --container-image="${CONTAINER_IMAGE}" \
    --container-mounts="${CONTAINER_MOUNTS}" \
    --container-workdir="${MEGATRON_ROOT}" \
    bash -c '
        set -euo pipefail
        export PYTHONPATH="${PYTHON_DEPS_DIR}:${PYTHONPATH:-}"
        export HF_HOME="${HF_HOME}"
        export HF_HUB_CACHE="${HF_HUB_CACHE}"
        if ! python -c "import hypercorn, quart" >/dev/null 2>&1; then
            echo "Installing quart==0.20.0 and its Hypercorn dependency."
            mkdir -p "${PYTHON_DEPS_DIR}"
            python -m pip install \
                --disable-pip-version-check \
                --upgrade \
                --target "${PYTHON_DEPS_DIR}" \
                "quart==0.20.0"
        fi
        python -c "import hypercorn, quart"
        echo "Caching official Hugging Face tokenizer ${TOKENIZER_MODEL}."
        python -c "import os; from transformers import AutoTokenizer; tokenizer = AutoTokenizer.from_pretrained(os.environ.get(\"TOKENIZER_MODEL\"), use_fast=True); assert tokenizer.is_fast, \"Official Ultra tokenizer did not resolve to a fast tokenizer\""
    '

SRUN_ARGS=(
    --nodes="${EXPECTED_NNODES}"
    --ntasks="${EXPECTED_NNODES}"
    --ntasks-per-node=1
    --gpus-per-node="${GPUS_PER_NODE}"
    --kill-on-bad-exit=1
    --container-image="${CONTAINER_IMAGE}"
    --container-mounts="${CONTAINER_MOUNTS}"
    --container-workdir="${MEGATRON_ROOT}"
)

srun "${SRUN_ARGS[@]}" bash "${WORKER_SCRIPT}" --worker &
SRUN_PID=$!

cleanup() {
    if kill -0 "${SRUN_PID}" 2>/dev/null; then
        kill -TERM "${SRUN_PID}" 2>/dev/null || true
        wait "${SRUN_PID}" 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM

health_url="http://${MASTER_ADDR}:${SERVER_PORT}/health"
started="${SECONDS}"
until curl --fail --silent --show-error "${health_url}" >/dev/null 2>&1; do
    if ! kill -0 "${SRUN_PID}" 2>/dev/null; then
        echo "Ultra aggregated server exited before becoming ready." >&2
        wait "${SRUN_PID}"
        exit 1
    fi
    if (( SECONDS - started >= STARTUP_TIMEOUT_SECONDS )); then
        echo "Timed out after ${STARTUP_TIMEOUT_SECONDS}s waiting for ${health_url}." >&2
        exit 1
    fi
    sleep 5
done

endpoint="http://${ENDPOINT_HOST}:${SERVER_PORT}/v1"
printf '%s\n' "${endpoint}" >"${ENDPOINT_FILE}"
echo "Ultra aggregated deployment is ready."
echo "OpenAI base URL: ${endpoint}"
echo "Endpoint file: ${ENDPOINT_FILE}"
echo "The endpoint remains live while SLURM job ${SLURM_JOB_ID} is running."

wait "${SRUN_PID}"
trap - EXIT INT TERM
