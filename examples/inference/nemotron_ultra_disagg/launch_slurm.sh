#!/usr/bin/env bash
# Four-node native disaggregated Nemotron Ultra BF16 deployment:
#   nodes 0-1: prefill, TP=8 / EP=8 / dense-DP=1
#   nodes 2-3: decode,  TP=8 / EP=8 / dense-DP=1
#
# Submit with:
#   sbatch --account=<account> --partition=<partition> launch_slurm.sh

#SBATCH --job-name=ultra-disagg
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --time=04:00:00
#SBATCH --output=ultra-disagg-%j.out

set -euo pipefail

export MEGATRON_ROOT="${MEGATRON_ROOT:-${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR is required}}"
export WORKER_SCRIPT="${MEGATRON_ROOT}/examples/inference/nemotron_ultra_disagg/launch_slurm.sh"
if [[ ! -f "${WORKER_SCRIPT}" ]]; then
    echo "Worker script is not visible at ${WORKER_SCRIPT}." >&2
    echo "Submit from the shared Megatron-LM worktree or export MEGATRON_ROOT explicitly." >&2
    exit 2
fi

export PRETRAINED_CHECKPOINT="${PRETRAINED_CHECKPOINT:-/lustre/fsw/portfolios/llmservice/projects/llmservice_nlp_fm/nemotron6/55b_hybrid_moe/checkpoints/pre_training_final_lc}"
export LOAD_CHECKPOINT="${LOAD_CHECKPOINT:-/lustre/fs1/portfolios/llmservice/projects/llmservice_fm_text/users/sasatheesh/data/checkpoints/inescapable-sawfly-step108-mcore}"
export TOKENIZER_MODEL="${TOKENIZER_MODEL:-/lustre/fsw/portfolios/llmservice/projects/llmservice_nlp_fm/nemotron6/tokenizers/multiMixV8.gpt4o_nc_sd.500000.128k.vocab.json}"
export CONTAINER_IMAGE="${CONTAINER_IMAGE:-/lustre/fsw/portfolios/nemotron/users/csathe/chaitrasathe+dynamo-megatron+mamba.sqsh}"
export CONTAINER_MOUNTS="${CONTAINER_MOUNTS:-/home:/home,/lustre:/lustre}"

if [[ ! -f "${CONTAINER_IMAGE}" ]]; then
    echo "Missing container image: ${CONTAINER_IMAGE}" >&2
    exit 2
fi

export GPUS_PER_NODE="${GPUS_PER_NODE:-4}"
export PREFILL_NODES=2
export DECODE_NODES=2
export EXPECTED_NNODES=$((PREFILL_NODES + DECODE_NODES))
export SHARD_WORLD_SIZE=$((PREFILL_NODES * GPUS_PER_NODE))
export TP_SIZE="${TP_SIZE:-8}"
export DENSE_DP_SIZE=$((SHARD_WORLD_SIZE / TP_SIZE))
export INFERENCE_SHARDS="tp=${TP_SIZE},pp=1,ep=${SHARD_WORLD_SIZE},expt_tp=1,dp=${DENSE_DP_SIZE},role=prefill+tp=${TP_SIZE},pp=1,ep=${SHARD_WORLD_SIZE},expt_tp=1,dp=${DENSE_DP_SIZE},role=decode"

export SERVER_PORT="${SERVER_PORT:-5000}"
export COORDINATOR_PORT="${COORDINATOR_PORT:-5555}"
export MASTER_PORT="${MASTER_PORT:-29500}"
export STARTUP_TIMEOUT_SECONDS="${STARTUP_TIMEOUT_SECONDS:-3600}"
export INFERENCE_BUFFER_SIZE_GB="${INFERENCE_BUFFER_SIZE_GB:-25}"
export INFERENCE_MAX_TOKENS="${INFERENCE_MAX_TOKENS:-8192}"
export INFERENCE_MAX_REQUESTS="${INFERENCE_MAX_REQUESTS:-32}"
export MAMBA_PREFIX_CACHE_GB="${MAMBA_PREFIX_CACHE_GB:-4}"

# GB200 does not require a CUDA_DEVICE_MAX_CONNECTIONS override.
export UCX_MEMTYPE_CACHE="${UCX_MEMTYPE_CACHE:-n}"

run_worker() {
    local node_rank="${SLURM_NODEID:?SLURM_NODEID is required}"

    cd "${MEGATRON_ROOT}"
    exec python -m torch.distributed.run \
        --nnodes="${EXPECTED_NNODES}" \
        --nproc-per-node="${GPUS_PER_NODE}" \
        --node-rank="${node_rank}" \
        --master-addr="${MASTER_ADDR}" \
        --master-port="${MASTER_PORT}" \
        -m examples.inference.launch_inference_server \
        --pretrained-checkpoint "${PRETRAINED_CHECKPOINT}" \
        --load "${LOAD_CHECKPOINT}" \
        --tokenizer-model "${TOKENIZER_MODEL}" \
        --model-provider mamba \
        --tensor-model-parallel-size "${TP_SIZE}" \
        --pipeline-model-parallel-size 1 \
        --expert-model-parallel-size "${SHARD_WORLD_SIZE}" \
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
        --seq-length 262144 \
        --max-position-embeddings 262144 \
        --inference-max-seq-length 262144 \
        --transformer-impl inference_optimized \
        --inference-grouped-gemm-backend vllm \
        --inference-use-synchronous-zmq-collectives \
        --mtp-use-repeated-layer \
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
        --inference-dynamic-batching-prefix-caching \
        --inference-dynamic-batching-prefix-caching-eviction-policy lru \
        --inference-dynamic-batching-prefix-caching-mamba-gb "${MAMBA_PREFIX_CACHE_GB}" \
        --inference-shards "${INFERENCE_SHARDS}" \
        --disagg-kv-transport-backend nixl \
        --coordinator-host "${MASTER_ADDR}" \
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
    echo "Run this script with sbatch (it requires four allocated nodes)." >&2
    exit 2
fi
if [[ "${SLURM_NNODES}" -ne "${EXPECTED_NNODES}" ]]; then
    echo "Expected ${EXPECTED_NNODES} nodes (2 prefill + 2 decode); got ${SLURM_NNODES}." >&2
    exit 2
fi

export MASTER_ADDR="${MASTER_ADDR:-$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n 1)}"
export ENDPOINT_HOST="${ENDPOINT_HOST:-${MASTER_ADDR}}"
export ENDPOINT_FILE="${ENDPOINT_FILE:-${MEGATRON_ROOT}/ultra-disagg-${SLURM_JOB_ID}.endpoint}"

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
        echo "Ultra disaggregated server exited before becoming ready." >&2
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
echo "Ultra disaggregated deployment is ready."
echo "OpenAI base URL: ${endpoint}"
echo "Endpoint file: ${ENDPOINT_FILE}"
echo "The endpoint remains live while SLURM job ${SLURM_JOB_ID} is running."

wait "${SRUN_PID}"
trap - EXIT INT TERM
