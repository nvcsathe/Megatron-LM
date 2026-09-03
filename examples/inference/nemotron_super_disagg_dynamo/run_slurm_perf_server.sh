#!/usr/bin/env bash
# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Persistent 24-GPU / six-node GB200 Dynamo deployment for Nemotron 3 Super:
#   nodes 0-3: four independent 4-GPU prefill workers
#              TP=1, PP=1, DP=4, ETP=1, EP=4, max batch=1
#   nodes 4-5: one 8-GPU decode worker spanning both nodes
#              TP=1, PP=1, DP=8, ETP=1, EP=8, max batch=376
#
# The batch script prints and writes an OpenAI-compatible endpoint, then stays
# alive until a service fails, the allocation expires, or the job is cancelled.

#SBATCH --job-name=super-dynamo-4p1d
#SBATCH --nodes=6
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --time=08:00:00
#SBATCH --output=slurm-super-dynamo-4p1d-%j.out

set -euo pipefail

export MEGATRON_ROOT="${MEGATRON_ROOT:-${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR is required}}"
export WORKER_SCRIPT="${MEGATRON_ROOT}/examples/inference/nemotron_super_disagg_dynamo/run_slurm_perf_server.sh"
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

export EXPECTED_NNODES=6
export GPUS_PER_NODE=4
export PREFILL_WORKERS=4
export PREFILL_TP_SIZE=1
export PREFILL_EP_SIZE=4
export PREFILL_ETP_SIZE=1
export PREFILL_MAX_REQUESTS="${PREFILL_MAX_REQUESTS:-1}"
export DECODE_WORKERS=1
export DECODE_NNODES=2
export DECODE_TP_SIZE=1
export DECODE_EP_SIZE=8
export DECODE_ETP_SIZE=1
export DECODE_MAX_REQUESTS="${DECODE_MAX_REQUESTS:-376}"

export INFERENCE_MAX_SEQ_LENGTH="${INFERENCE_MAX_SEQ_LENGTH:-8192}"
export INFERENCE_BUFFER_SIZE_GB="${INFERENCE_BUFFER_SIZE_GB:-20}"
export INFERENCE_MAX_TOKENS="${INFERENCE_MAX_TOKENS:-8192}"
export STARTUP_TIMEOUT_SECONDS="${STARTUP_TIMEOUT_SECONDS:-3600}"
export RUN_BASE_DIR="${RUN_BASE_DIR:-${MEGATRON_ROOT}/logs}"
export RUN_DIR="${RUN_BASE_DIR}/nemotron-super-dynamo-4p1d-${SLURM_JOB_ID:-manual}"
export ENDPOINT_FILE="${RUN_DIR}/endpoint.env"
export CONTROL_STATE_DIR="${CONTROL_STATE_DIR:-/tmp/nemotron-super-dynamo-4p1d-${SLURM_JOB_ID:-manual}}"

# Allocate a job-specific block. Worker-local coordinator/system ports could be
# reused across hosts, but keeping them distinct makes logs and overrides clear.
export JOB_PORT_BASE="${JOB_PORT_BASE:-31000}"
export HTTP_PORT="${HTTP_PORT:-${JOB_PORT_BASE}}"
export ETCD_PORT="${ETCD_PORT:-$((JOB_PORT_BASE + 1))}"
export ETCD_PEER_PORT="${ETCD_PEER_PORT:-$((JOB_PORT_BASE + 2))}"
export NATS_PORT="${NATS_PORT:-$((JOB_PORT_BASE + 3))}"
export FRONTEND_SYSTEM_PORT="${FRONTEND_SYSTEM_PORT:-$((JOB_PORT_BASE + 4))}"
export PREFILL_SYSTEM_PORT_BASE="${PREFILL_SYSTEM_PORT_BASE:-$((JOB_PORT_BASE + 10))}"
export PREFILL_COORDINATOR_PORT_BASE="${PREFILL_COORDINATOR_PORT_BASE:-$((JOB_PORT_BASE + 20))}"
export DECODE_SYSTEM_PORT="${DECODE_SYSTEM_PORT:-$((JOB_PORT_BASE + 30))}"
export DECODE_COORDINATOR_PORT="${DECODE_COORDINATOR_PORT:-$((JOB_PORT_BASE + 31))}"
export DECODE_MASTER_PORT="${DECODE_MASTER_PORT:-$((JOB_PORT_BASE + 32))}"

# Keep the transport defaults aligned with the validated cross-node smoke test.
# Override these for a production RDMA/NIXL setup when benchmarking transport.
export UCX_TLS="${UCX_TLS_OVERRIDE:-cuda_copy,tcp,shm,cma,self}"
export UCX_MEMTYPE_CACHE="${UCX_MEMTYPE_CACHE_OVERRIDE:-n}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
# The decode Dynamo parent launches its two-node engine through a nested srun.
# Slurm and Pyxis consume these environment variables as the equivalent of
# --overlap and the corresponding --container-* options, so the nested step
# needs no changes to the Megatron Dynamo integration.
export SLURM_OVERLAP=1
export PYXIS_CONTAINER_IMAGE="${CONTAINER_IMAGE}"
export PYXIS_CONTAINER_MOUNTS="${CONTAINER_MOUNTS}"
export PYXIS_CONTAINER_WORKDIR="${MEGATRON_ROOT}"
# CUDA_DEVICE_MAX_CONNECTIONS is intentionally unset for GB200.

prefill_worker_file() {
    local index="$1"
    printf '%s/prefill-worker-%s.json\n' "${RUN_DIR}" "${index}"
}

decode_worker_file() {
    printf '%s/decode-worker-0.json\n' "${RUN_DIR}"
}

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
    local role="$1"
    local tp_size ep_size etp_size max_requests
    case "${role}" in
        prefill)
            tp_size="${PREFILL_TP_SIZE}"
            ep_size="${PREFILL_EP_SIZE}"
            etp_size="${PREFILL_ETP_SIZE}"
            max_requests="${PREFILL_MAX_REQUESTS}"
            ;;
        decode)
            tp_size="${DECODE_TP_SIZE}"
            ep_size="${DECODE_EP_SIZE}"
            etp_size="${DECODE_ETP_SIZE}"
            max_requests="${DECODE_MAX_REQUESTS}"
            ;;
        *)
            echo "Unknown role: ${role}" >&2
            return 2
            ;;
    esac

    cat <<EOF
--tensor-model-parallel-size
${tp_size}
--expert-tensor-parallel-size
${etp_size}
--expert-model-parallel-size
${ep_size}
--pipeline-model-parallel-size
1
--model-provider
mamba
--inference-max-seq-length
${INFERENCE_MAX_SEQ_LENGTH}
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
${INFERENCE_BUFFER_SIZE_GB}
--inference-dynamic-batching-max-tokens
${INFERENCE_MAX_TOKENS}
--enable-chunked-prefill
--inference-dynamic-batching-prefix-caching
--inference-logging-step-interval
100
--inference-dynamic-batching-num-cuda-graphs
-1
--inference-cuda-graph-scope
block
--inference-dynamic-batching-max-requests
${max_requests}
--num-speculative-tokens
2
--distributed-backend
nccl
--disagg-kv-transport-backend
nixl
EOF
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

run_control_plane() {
    CONTROL_CHILD_PIDS=()
    trap cleanup_control_children EXIT INT TERM
    mkdir -p "${CONTROL_STATE_DIR}/etcd" "${CONTROL_STATE_DIR}/nats"

    etcd \
        --name "nemotron-super-4p1d-${SLURM_JOB_ID}" \
        --data-dir "${CONTROL_STATE_DIR}/etcd" \
        --listen-client-urls "http://0.0.0.0:${ETCD_PORT}" \
        --advertise-client-urls "http://${CONTROL_HOST}:${ETCD_PORT}" \
        --listen-peer-urls "http://0.0.0.0:${ETCD_PEER_PORT}" \
        --initial-advertise-peer-urls "http://${CONTROL_HOST}:${ETCD_PEER_PORT}" \
        --initial-cluster "nemotron-super-4p1d-${SLURM_JOB_ID}=http://${CONTROL_HOST}:${ETCD_PEER_PORT}" \
        --initial-cluster-token "nemotron-super-4p1d-${SLURM_JOB_ID}" \
        >"${RUN_DIR}/etcd.log" 2>&1 &
    CONTROL_CHILD_PIDS+=("$!")

    nats-server -js --name "nemotron-super-4p1d-${SLURM_JOB_ID}" \
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

    set +e
    wait -n "${CONTROL_CHILD_PIDS[@]}"
    local rc=$?
    set -e
    echo "A control-plane service exited unexpectedly (status ${rc})." >&2
    return 1
}

run_prefill_worker() {
    local index="${1:?prefill worker index is required}"
    local worker_file system_port coordinator_port coordinator_host
    worker_file="$(prefill_worker_file "${index}")"
    system_port=$((PREFILL_SYSTEM_PORT_BASE + index))
    coordinator_port=$((PREFILL_COORDINATOR_PORT_BASE + index))
    coordinator_host="$(hostname -f)"
    local -a model_args=()
    mapfile -t model_args < <(megatron_args prefill)

    wait_for_control_plane
    cd "${MEGATRON_ROOT}"
    DYN_SYSTEM_PORT="${system_port}" exec python -m megatron.inference.integrations.dynamo \
        --role prefill \
        --component prefill \
        --model "${TOKENIZER_MODEL}" \
        --served-model-name "${SERVED_MODEL_NAME}" \
        --endpoint-types completions \
        --nproc-per-node "${GPUS_PER_NODE}" \
        --coordinator-host "${coordinator_host}" \
        --coordinator-port "${coordinator_port}" \
        --worker-id-file "${worker_file}" \
        --megatron-root "${MEGATRON_ROOT}" \
        --discovery-backend etcd \
        --request-plane nats \
        --event-plane nats \
        -- "${model_args[@]}"
}

run_decode_worker() {
    local worker_file
    worker_file="$(decode_worker_file)"
    local -a model_args=()
    mapfile -t model_args < <(megatron_args decode)

    wait_for_control_plane
    cd "${MEGATRON_ROOT}"
    DYN_SYSTEM_PORT="${DECODE_SYSTEM_PORT}" exec python -m megatron.inference.integrations.dynamo \
        --role decode \
        --component backend \
        --model "${TOKENIZER_MODEL}" \
        --served-model-name "${SERVED_MODEL_NAME}" \
        --endpoint-types completions \
        --launcher slurm \
        --nnodes "${DECODE_NNODES}" \
        --nproc-per-node "${GPUS_PER_NODE}" \
        --master-addr "${DECODE_LEADER_HOST}" \
        --master-port "${DECODE_MASTER_PORT}" \
        --slurm-nodelist "${DECODE_NODELIST}" \
        --parent-event-host "${DECODE_LEADER_HOST}" \
        --coordinator-host "${DECODE_LEADER_HOST}" \
        --coordinator-port "${DECODE_COORDINATOR_PORT}" \
        --worker-id-file "${worker_file}" \
        --megatron-root "${MEGATRON_ROOT}" \
        --discovery-backend etcd \
        --request-plane nats \
        --event-plane nats \
        -- "${model_args[@]}"
}

case "${1:-}" in
    --control-plane)
        run_control_plane
        exit
        ;;
    --prefill-worker)
        run_prefill_worker "${2:-}"
        exit
        ;;
    --decode-worker)
        run_decode_worker
        exit
        ;;
    "")
        ;;
    *)
        echo "Unknown launcher mode: $1" >&2
        exit 2
        ;;
esac

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    echo "Run this server with sbatch; it requires six allocated nodes." >&2
    exit 2
fi
if [[ "${SLURM_NNODES}" -ne "${EXPECTED_NNODES}" ]]; then
    echo "Expected ${EXPECTED_NNODES} nodes; got ${SLURM_NNODES}." >&2
    exit 2
fi
if (( JOB_PORT_BASE < 1024 || JOB_PORT_BASE + 32 > 65535 )); then
    echo "JOB_PORT_BASE must leave room for ports through JOB_PORT_BASE+32." >&2
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

mapfile -t allocated_nodes < <(scontrol show hostnames "${SLURM_JOB_NODELIST}")
if [[ "${#allocated_nodes[@]}" -ne "${EXPECTED_NNODES}" ]]; then
    echo "Expected ${EXPECTED_NNODES} resolved hosts; got ${#allocated_nodes[@]}." >&2
    exit 2
fi

export CONTROL_HOST="${CONTROL_HOST:-${allocated_nodes[0]}}"
export DECODE_LEADER_HOST="${allocated_nodes[4]}"
export DECODE_NODELIST="${allocated_nodes[4]},${allocated_nodes[5]}"
export ETCD_ENDPOINTS="http://${CONTROL_HOST}:${ETCD_PORT}"
export NATS_SERVER="nats://${CONTROL_HOST}:${NATS_PORT}"

mkdir -p "${RUN_DIR}"
rm -f "${ENDPOINT_FILE}" "$(decode_worker_file)"
for ((index = 0; index < PREFILL_WORKERS; index++)); do
    rm -f "$(prefill_worker_file "${index}")"
done

echo "Nemotron Super Dynamo 4P1D deployment ${SLURM_JOB_ID}"
echo "  control/frontend: ${CONTROL_HOST}"
for ((index = 0; index < PREFILL_WORKERS; index++)); do
    echo "  prefill ${index}: ${allocated_nodes[index]} (4 GPUs, TP1/DP4/ETP1/EP4)"
done
echo "  decode 0: ${DECODE_NODELIST} (8 GPUs, TP1/DP8/ETP1/EP8)"
echo "  shared logs: ${RUN_DIR}"

CONTAINER_ARGS=(
    --container-image="${CONTAINER_IMAGE}"
    --container-mounts="${CONTAINER_MOUNTS}"
    --container-workdir="${MEGATRON_ROOT}"
)
declare -a SERVICE_PIDS=()
declare -a SERVICE_NAMES=()

launch_service() {
    local name="$1"
    shift
    "$@" >"${RUN_DIR}/${name}.log" 2>&1 &
    SERVICE_PIDS+=("$!")
    SERVICE_NAMES+=("${name}")
}

cleanup() {
    trap - EXIT INT TERM
    local pid
    for pid in "${SERVICE_PIDS[@]:-}"; do
        [[ -n "${pid}" ]] || continue
        kill -TERM "${pid}" 2>/dev/null || true
    done
    for pid in "${SERVICE_PIDS[@]:-}"; do
        [[ -n "${pid}" ]] || continue
        wait "${pid}" 2>/dev/null || true
    done
}
trap cleanup EXIT INT TERM

launch_service control-plane \
    srun --overlap --nodes=1 --ntasks=1 --nodelist="${CONTROL_HOST}" \
    "${CONTAINER_ARGS[@]}" bash "${WORKER_SCRIPT}" --control-plane
wait_for_control_plane

for ((index = 0; index < PREFILL_WORKERS; index++)); do
    launch_service "prefill-${index}" \
        srun --overlap --nodes=1 --ntasks=1 --nodelist="${allocated_nodes[index]}" \
        --gpus-per-node="${GPUS_PER_NODE}" --kill-on-bad-exit=1 \
        "${CONTAINER_ARGS[@]}" bash "${WORKER_SCRIPT}" --prefill-worker "${index}"
done

# The lightweight Dynamo parent shares decode node 0 with the engine job step.
# SLURM_OVERLAP and the PYXIS_CONTAINER_* variables apply to its child srun.
launch_service decode-0 \
    srun --overlap --nodes=1 --ntasks=1 --nodelist="${DECODE_LEADER_HOST}" \
    "${CONTAINER_ARGS[@]}" bash "${WORKER_SCRIPT}" --decode-worker

service_failed() {
    local index pid rc
    for index in "${!SERVICE_PIDS[@]}"; do
        pid="${SERVICE_PIDS[index]}"
        if ! kill -0 "${pid}" 2>/dev/null; then
            set +e
            wait "${pid}"
            rc=$?
            set -e
            echo "Service ${SERVICE_NAMES[index]} exited (status ${rc})." >&2
            return 0
        fi
    done
    return 1
}

declare -a READY_FILES=("$(decode_worker_file)")
for ((index = 0; index < PREFILL_WORKERS; index++)); do
    READY_FILES+=("$(prefill_worker_file "${index}")")
done

started="${SECONDS}"
while true; do
    all_ready=1
    for ready_file in "${READY_FILES[@]}"; do
        if [[ ! -s "${ready_file}" ]]; then
            all_ready=0
            break
        fi
    done
    (( all_ready == 1 )) && break
    if service_failed; then
        exit 1
    fi
    if (( SECONDS - started >= STARTUP_TIMEOUT_SECONDS )); then
        echo "Timed out waiting for all ${PREFILL_WORKERS} prefill and ${DECODE_WORKERS} decode workers." >&2
        exit 1
    fi
    sleep 5
done

wait_for_url "http://${CONTROL_HOST}:${HTTP_PORT}/v1/models" "Dynamo frontend"

endpoint_url="http://${CONTROL_HOST}:${HTTP_PORT}/v1"
endpoint_tmp="${ENDPOINT_FILE}.tmp"
printf 'export DYNAMO_ENDPOINT=%q\nexport DYNAMO_MODEL=%q\nexport DYNAMO_RUN_DIR=%q\n' \
    "${endpoint_url}" "${SERVED_MODEL_NAME}" "${RUN_DIR}" >"${endpoint_tmp}"
mv "${endpoint_tmp}" "${ENDPOINT_FILE}"

echo "Nemotron Super Dynamo 4P1D server ready."
echo "  endpoint: ${endpoint_url}"
echo "  model: ${SERVED_MODEL_NAME}"
echo "  endpoint env: ${ENDPOINT_FILE}"
echo "  stop with: scancel ${SLURM_JOB_ID}"

while true; do
    if service_failed; then
        exit 1
    fi
    sleep 5
done
