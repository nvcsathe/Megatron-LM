#!/usr/bin/env bash
# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Two-node GB200 Dynamo disaggregated Nemotron Super BF16 smoke test:
#   node 0: Dynamo frontend and one 4-GPU prefill replica (TP=2, EP=4)
#   node 1: one 4-GPU decode replica (TP=2, EP=4)

#SBATCH --job-name=super-dynamo-disagg
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --time=02:00:00
#SBATCH --output=slurm-super-dynamo-disagg-%j.out

set -euo pipefail

export MEGATRON_ROOT="${MEGATRON_ROOT:-${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR is required}}"
export WORKER_SCRIPT="${MEGATRON_ROOT}/examples/inference/nemotron_super_disagg_dynamo/run_slurm_test.sh"
if [[ ! -f "${WORKER_SCRIPT}" ]]; then
    echo "Worker script is not visible at ${WORKER_SCRIPT}." >&2
    echo "Submit from the shared Megatron-LM worktree or export MEGATRON_ROOT." >&2
    exit 2
fi

export LOAD_CHECKPOINT="${LOAD_CHECKPOINT:-/lustre/fsw/portfolios/nemotron/projects/nemotron_sw_pre/users/ksanthanam/nemotron-3-super-120b-a12b}"
export TOKENIZER_MODEL="${TOKENIZER_MODEL:-nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16}"
export SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-${TOKENIZER_MODEL}}"
export CONTAINER_IMAGE="${CONTAINER_IMAGE:-/lustre/fsw/portfolios/nemotron/users/csathe/chaitrasathe+dynamo-megatron+mamba-dynamo-1.3.1.sqsh}"
export CONTAINER_MOUNTS="${CONTAINER_MOUNTS:-/home:/home,/lustre:/lustre}"

export EXPECTED_NNODES=2
export GPUS_PER_NODE="${GPUS_PER_NODE:-4}"
export TP_SIZE="${TP_SIZE:-2}"
export EP_SIZE="${EP_SIZE:-4}"
export ETP_SIZE="${ETP_SIZE:-1}"
if (( GPUS_PER_NODE != 4 || TP_SIZE != 2 || EP_SIZE != 4 || ETP_SIZE != 1 )); then
    echo "This profile requires 4 GPUs per replica, TP=2, EP=4, and expert-TP=1." >&2
    echo "Got GPUs=${GPUS_PER_NODE}, TP=${TP_SIZE}, EP=${EP_SIZE}, expert-TP=${ETP_SIZE}." >&2
    exit 2
fi
if [[ ! -r "${CONTAINER_IMAGE}" ]]; then
    echo "Container image is not readable: ${CONTAINER_IMAGE}" >&2
    exit 2
fi
if [[ ! -d "${LOAD_CHECKPOINT}" ]]; then
    echo "Checkpoint directory is not visible: ${LOAD_CHECKPOINT}" >&2
    exit 2
fi

# Give every batch job its own block of ports. With ten ports per block, the
# computed defaults range from 20000 through 59997 and remain valid TCP ports.
export JOB_PORT_BASE="${JOB_PORT_BASE:-$((20000 + (10#${SLURM_JOB_ID} % 4000) * 10))}"
export HTTP_PORT="${HTTP_PORT:-${JOB_PORT_BASE}}"
export ETCD_PORT="${ETCD_PORT:-$((JOB_PORT_BASE + 1))}"
export ETCD_PEER_PORT="${ETCD_PEER_PORT:-$((JOB_PORT_BASE + 2))}"
export NATS_PORT="${NATS_PORT:-$((JOB_PORT_BASE + 3))}"
export COORDINATOR_PORT="${COORDINATOR_PORT:-$((JOB_PORT_BASE + 4))}"
export FRONTEND_SYSTEM_PORT="${FRONTEND_SYSTEM_PORT:-$((JOB_PORT_BASE + 5))}"
export PREFILL_SYSTEM_PORT="${PREFILL_SYSTEM_PORT:-$((JOB_PORT_BASE + 6))}"
export DECODE_SYSTEM_PORT="${DECODE_SYSTEM_PORT:-$((JOB_PORT_BASE + 7))}"
export STARTUP_TIMEOUT_SECONDS="${STARTUP_TIMEOUT_SECONDS:-3600}"
export REQUEST_TIMEOUT_SECONDS="${REQUEST_TIMEOUT_SECONDS:-300}"
export RUN_DIR="${RUN_DIR:-${MEGATRON_ROOT}/logs/nemotron-super-dynamo-disagg-${SLURM_JOB_ID:-manual}}"
export PREFILL_WORKER_FILE="${RUN_DIR}/prefill-worker.json"
export DECODE_WORKER_FILE="${RUN_DIR}/decode-worker.json"
# etcd and NATS databases are mutable service state and must not live on
# Lustre. /tmp is node-local inside the Pyxis container, and the job ID makes
# the path unique even if state from an earlier container survives briefly.
export CONTROL_STATE_DIR="${CONTROL_STATE_DIR:-/tmp/nemotron-super-dynamo-disagg-${SLURM_JOB_ID}}"

# Dynamo uses these addresses for cross-node discovery and messaging.
export ETCD_ENDPOINTS="http://${CONTROL_HOST:-127.0.0.1}:${ETCD_PORT}"
export NATS_SERVER="nats://${CONTROL_HOST:-127.0.0.1}:${NATS_PORT}"
# Use Dynamo's explicit NIXL/InfiniBand transport set. Assign this
# unconditionally because the container image may define UCX_TLS and thereby
# defeat a shell-default exclusion of the gdr_copy memory domain.
export UCX_TLS="rc_x,rc,cuda_copy,cuda_ipc"
export UCX_MEMTYPE_CACHE="${UCX_MEMTYPE_CACHE:-n}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
# CUDA_DEVICE_MAX_CONNECTIONS is intentionally left unset for GB200. Export it
# as 1 before submission when adapting this non-FSDP TP profile to Hopper.

wait_for_url() {
    local url="$1"
    local description="$2"
    local started="${SECONDS}"
    until curl --fail --silent --show-error "${url}" >/dev/null 2>&1; do
        if (( SECONDS - started >= STARTUP_TIMEOUT_SECONDS )); then
            echo "Timed out waiting for ${description} at ${url}." >&2
            return 1
        fi
        sleep 2
    done
}

wait_for_control_plane() {
    wait_for_url "http://${CONTROL_HOST}:${ETCD_PORT}/health" "etcd"
    local started="${SECONDS}"
    until (exec 3<>"/dev/tcp/${CONTROL_HOST}/${NATS_PORT}") 2>/dev/null; do
        if (( SECONDS - started >= STARTUP_TIMEOUT_SECONDS )); then
            echo "Timed out waiting for NATS at ${CONTROL_HOST}:${NATS_PORT}." >&2
            return 1
        fi
        sleep 2
    done
}

megatron_args() {
    cat <<EOF
--tensor-model-parallel-size
${TP_SIZE}
--expert-tensor-parallel-size
${ETP_SIZE}
--expert-model-parallel-size
${EP_SIZE}
--sequence-parallel
--pipeline-model-parallel-size
1
--model-provider
mamba
--inference-max-seq-length
8192
--mtp-use-repeated-layer
--load
${LOAD_CHECKPOINT}
--micro-batch-size
1
--moe-router-dtype
fp32
--moe-token-dispatcher-type
alltoall
--use-checkpoint-args
--bf16
--attention-backend
flash
--transformer-impl
inference_optimized
--no-load-optim
--te-rng-tracker
--inference-rng-tracker
--cuda-graph-impl
local
--dist-ckpt-strictness
log_unexpected
--no-gradient-accumulation-fusion
--hidden-size
4096
--ffn-hidden-size
2688
--seq-length
1048576
--num-attention-heads
32
--num-query-groups
2
--group-query-attention
--kv-channels
128
--max-position-embeddings
1048576
--position-embedding-type
none
--rotary-base
10000
--rotary-percent
1.0
--disable-bias-linear
--squared-relu
--untie-embeddings-and-output-weights
--normalization
RMSNorm
--attention-dropout
0.0
--hidden-dropout
0.0
--mtp-hybrid-override-pattern
none
--mtp-num-layers
2
--spec
megatron.core.models.mamba.mamba_layer_specs
mamba_stack_spec
--num-experts
512
--moe-layer-freq
1
--moe-ffn-hidden-size
2688
--moe-router-topk
22
--moe-grouped-gemm
--moe-shared-expert-intermediate-size
5376
--moe-router-score-function
sigmoid
--moe-router-enable-expert-bias
--moe-router-topk-scaling-factor
5.0
--mamba-state-dim
128
--mamba-head-dim
64
--mamba-num-groups
8
--mamba-num-heads
128
--hybrid-layer-pattern
MEMEMEM*EMEMEMEM*EMEMEMEM*EMEMEMEMEM*EMEMEMEMEM*EMEMEMEMEM*EMEMEMEMEM*EMEMEMEM*EMEMEMEME/*E/*E
--moe-latent-size
1024
--padded-vocab-size
131072
--tokenizer-type
HuggingFaceTokenizer
--tokenizer-model
${TOKENIZER_MODEL}
--no-use-tokenizer-model-from-checkpoint-args
--inference-dynamic-batching
--inference-dynamic-batching-buffer-size-gb
20
--inference-dynamic-batching-max-tokens
8192
--enable-chunked-prefill
--inference-dynamic-batching-prefix-caching
--inference-logging-step-interval
100
--inference-dynamic-batching-num-cuda-graphs
-1
--inference-cuda-graph-scope
block
--inference-dynamic-batching-max-requests
64
--num-speculative-tokens
2
--distributed-backend
nccl
--disagg-kv-transport-backend
nixl
EOF
}

run_dynamo_worker() {
    local role="$1"
    local component="$2"
    local worker_file="$3"
    local system_port="$4"
    local coordinator_host
    coordinator_host="$(hostname -f)"
    local -a model_args=()
    mapfile -t model_args < <(megatron_args)

    cd "${MEGATRON_ROOT}"
    DYN_SYSTEM_PORT="${system_port}" exec python -m megatron.inference.integrations.dynamo \
        --role "${role}" \
        --component "${component}" \
        --model "${TOKENIZER_MODEL}" \
        --served-model-name "${SERVED_MODEL_NAME}" \
        --endpoint-types completions \
        --nproc-per-node "${GPUS_PER_NODE}" \
        --coordinator-host "${coordinator_host}" \
        --coordinator-port "${COORDINATOR_PORT}" \
        --worker-id-file "${worker_file}" \
        --megatron-root "${MEGATRON_ROOT}" \
        --discovery-backend etcd \
        --request-plane nats \
        --event-plane nats \
        -- "${model_args[@]}"
}

declare -a CONTROL_CHILD_PIDS=()

cleanup_control_children() {
    trap - EXIT INT TERM
    local pid
    for pid in "${CONTROL_CHILD_PIDS[@]:-}"; do
        [[ -n "${pid}" ]] || continue
        kill -TERM "${pid}" 2>/dev/null || true
    done
    for pid in "${CONTROL_CHILD_PIDS[@]:-}"; do
        [[ -n "${pid}" ]] || continue
        wait "${pid}" 2>/dev/null || true
    done
    CONTROL_CHILD_PIDS=()
}

run_control_node() {
    CONTROL_CHILD_PIDS=()
    trap cleanup_control_children EXIT INT TERM
    mkdir -p "${CONTROL_STATE_DIR}/etcd" "${CONTROL_STATE_DIR}/nats"

    etcd \
        --name "nemotron-super-disagg-${SLURM_JOB_ID}" \
        --data-dir "${CONTROL_STATE_DIR}/etcd" \
        --listen-client-urls "http://0.0.0.0:${ETCD_PORT}" \
        --advertise-client-urls "http://${CONTROL_HOST}:${ETCD_PORT}" \
        --listen-peer-urls "http://0.0.0.0:${ETCD_PEER_PORT}" \
        --initial-advertise-peer-urls "http://${CONTROL_HOST}:${ETCD_PEER_PORT}" \
        --initial-cluster "nemotron-super-disagg-${SLURM_JOB_ID}=http://${CONTROL_HOST}:${ETCD_PEER_PORT}" \
        --initial-cluster-token "nemotron-super-disagg-${SLURM_JOB_ID}" \
        >"${RUN_DIR}/etcd.log" 2>&1 &
    CONTROL_CHILD_PIDS+=("$!")

    nats-server -js --name "nemotron-super-disagg-${SLURM_JOB_ID}" \
        -a 0.0.0.0 -p "${NATS_PORT}" \
        --store_dir "${CONTROL_STATE_DIR}/nats" >"${RUN_DIR}/nats.log" 2>&1 &
    CONTROL_CHILD_PIDS+=("$!")

    wait_for_control_plane
    DYN_SYSTEM_PORT="${FRONTEND_SYSTEM_PORT}" python -m dynamo.frontend \
        --http-port "${HTTP_PORT}" \
        --router-mode kv \
        --router-min-initial-workers 1 \
        --discovery-backend etcd \
        --request-plane nats \
        --event-plane nats \
        >"${RUN_DIR}/frontend.log" 2>&1 &
    CONTROL_CHILD_PIDS+=("$!")

    run_dynamo_worker prefill prefill "${PREFILL_WORKER_FILE}" "${PREFILL_SYSTEM_PORT}" \
        >"${RUN_DIR}/prefill.log" 2>&1 &
    CONTROL_CHILD_PIDS+=("$!")

    set +e
    wait -n "${CONTROL_CHILD_PIDS[@]}"
    local rc=$?
    set -e
    echo "A control-node service exited unexpectedly (status ${rc})." >&2
    return 1
}

run_decode_node() {
    wait_for_control_plane
    run_dynamo_worker decode backend "${DECODE_WORKER_FILE}" "${DECODE_SYSTEM_PORT}" \
        >"${RUN_DIR}/decode.log" 2>&1
}

if [[ "${1:-}" == "--worker" ]]; then
    echo "SLURM task ${SLURM_PROCID:-unknown}: UCX_TLS=${UCX_TLS}, UCX_MEMTYPE_CACHE=${UCX_MEMTYPE_CACHE}"
    case "${SLURM_PROCID:?SLURM_PROCID is required}" in
        0) run_control_node ;;
        1) run_decode_node ;;
        *) echo "Unexpected SLURM process rank ${SLURM_PROCID}." >&2; exit 2 ;;
    esac
    exit
fi

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    echo "Run this test with sbatch; it requires two allocated nodes." >&2
    exit 2
fi
if [[ "${SLURM_NNODES}" -ne "${EXPECTED_NNODES}" ]]; then
    echo "Expected two nodes (one prefill and one decode); got ${SLURM_NNODES}." >&2
    exit 2
fi

mapfile -t allocated_nodes < <(scontrol show hostnames "${SLURM_JOB_NODELIST}")
export CONTROL_HOST="${CONTROL_HOST:-${allocated_nodes[0]}}"
export ETCD_ENDPOINTS="http://${CONTROL_HOST}:${ETCD_PORT}"
export NATS_SERVER="nats://${CONTROL_HOST}:${NATS_PORT}"
mkdir -p "${RUN_DIR}"
rm -f "${PREFILL_WORKER_FILE}" "${DECODE_WORKER_FILE}"

echo "Nemotron Super Dynamo job ${SLURM_JOB_ID}"
echo "  control host: ${CONTROL_HOST}"
echo "  HTTP/etcd/NATS/coordinator ports: ${HTTP_PORT}/${ETCD_PORT}/${NATS_PORT}/${COORDINATOR_PORT}"
echo "  Dynamo system ports: ${FRONTEND_SYSTEM_PORT}/${PREFILL_SYSTEM_PORT}/${DECODE_SYSTEM_PORT}"
echo "  node-local control state: ${CONTROL_STATE_DIR}"
echo "  shared logs: ${RUN_DIR}"

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
    trap - EXIT INT TERM
    if kill -0 "${SRUN_PID}" 2>/dev/null; then
        kill -TERM "${SRUN_PID}" 2>/dev/null || true
        wait "${SRUN_PID}" 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM

started="${SECONDS}"
while [[ ! -s "${PREFILL_WORKER_FILE}" || ! -s "${DECODE_WORKER_FILE}" ]]; do
    if ! kill -0 "${SRUN_PID}" 2>/dev/null; then
        echo "Dynamo deployment exited before both workers registered." >&2
        wait "${SRUN_PID}"
        exit 1
    fi
    if (( SECONDS - started >= STARTUP_TIMEOUT_SECONDS )); then
        echo "Timed out waiting for prefill and decode Dynamo registrations." >&2
        exit 1
    fi
    sleep 5
done

wait_for_url "http://${CONTROL_HOST}:${HTTP_PORT}/v1/models" "Dynamo frontend"

response_file="${RUN_DIR}/completion-response.json"
curl --fail --silent --show-error \
    --max-time "${REQUEST_TIMEOUT_SECONDS}" \
    "http://${CONTROL_HOST}:${HTTP_PORT}/v1/completions" \
    -H 'Content-Type: application/json' \
    -d "{\"model\":\"${SERVED_MODEL_NAME}\",\"prompt\":\"Explain disaggregated inference in one sentence.\",\"max_tokens\":32,\"temperature\":0}" \
    -o "${response_file}"

python -c 'import json, sys; data = json.load(open(sys.argv[1])); choices = data.get("choices") or []; assert choices and choices[0].get("text"), data; print("PASS:", choices[0]["text"])' "${response_file}"
echo "Nemotron Super Dynamo disaggregated smoke test passed. Logs: ${RUN_DIR}"

cleanup
trap - EXIT INT TERM
