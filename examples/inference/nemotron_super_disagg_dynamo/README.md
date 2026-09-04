# Nemotron Super disaggregated inference with Dynamo

This is a two-node GB200 SLURM smoke test for Nemotron 3 Super BF16 using the
Megatron-owned Dynamo backend:

- node 0 runs etcd, NATS, the Dynamo KV-routing frontend, and one four-GPU
  prefill replica;
- node 1 runs one four-GPU decode replica;
- each replica preserves the aggregated profile's TP=2, EP=4, expert-TP=1,
  and PP=1 parallelism; and
- NIXL transfers KV and Mamba state from the prefill replica to the decode
  replica.

Prefill and decode are separate Dynamo workers and independent one-node
`torch.distributed` groups. They are deliberately not combined into one
eight-rank Megatron world: each worker must own a complete four-GPU model
replica.

## Run the smoke test

Use a container that contains Megatron-LM's Dynamo dependencies, NIXL, `etcd`,
and `nats-server`. The worktree and checkpoint must be visible at the same path
on both nodes.

```bash
cd /path/to/Megatron-LM
sbatch --account=<account> --partition=<partition> \
  examples/inference/nemotron_super_disagg_dynamo/run_slurm_test.sh
```

The script is self-contained for the Nemotron cluster environment. Its default
container, checkpoint, and tokenizer are:

```text
/lustre/fsw/portfolios/nemotron/users/csathe/chaitrasathe+dynamo-megatron+mamba-dynamo-1.3.1.sqsh
/lustre/fsw/portfolios/nemotron/projects/nemotron_sw_pre/users/ksanthanam/nemotron-3-super-120b-a12b
nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16
```

The launcher starts the Dynamo control plane and both workers, waits for their
registrations, sends one deterministic `/v1/completions` request, validates
that the response contains generated text, and shuts everything down. Logs and
the response are retained under
`${RUN_BASE_DIR:-${MEGATRON_ROOT}/logs}/nemotron-super-dynamo-disagg-${SLURM_JOB_ID}`
for diagnosis. The job-specific suffix is always derived from the current
allocation so an inherited path cannot mix artifacts from different jobs.

## Run the 24-GPU 4P1D performance server

`run_slurm_perf_server.sh` deploys the AIConfigurator 4P1D topology on six
four-GPU GB200 nodes and leaves it running for a separate benchmark job:

- nodes 0-3 each run one four-GPU prefill worker with
  `tp1pp1dp4etp1ep4` and a maximum batch size of 1;
- nodes 4-5 together run one eight-GPU decode worker with
  `tp1pp1dp8etp1ep8` and a maximum batch size of 376; and
- the complete deployment uses 24 GPUs. The decode worker is one distributed
  Megatron rank group, not two independent workers.

Submit it from the shared worktree:

```bash
JOB_ID=$(sbatch --parsable --account=<account> --partition=<partition> \
  examples/inference/nemotron_super_disagg_dynamo/run_slurm_perf_server.sh)
```

When all five Dynamo workers are registered, the job log prints the endpoint
and the launcher writes a sourceable endpoint file:

```bash
ENDPOINT_FILE="${RUN_BASE_DIR:-${MEGATRON_ROOT:-$PWD}/logs}/nemotron-super-dynamo-4p1d-${JOB_ID}/endpoint.env"
source "$ENDPOINT_FILE"
curl "$DYNAMO_ENDPOINT/models"
```

The endpoint is routable from compute nodes on the same cluster network. Point
the separate benchmark job at `DYNAMO_ENDPOINT` and use `DYNAMO_MODEL` as the
OpenAI model name. This launcher does not generate traffic itself. Stop the
deployment with `scancel "$JOB_ID"`.

The default inference limits reproduce the planner row: prefill batch 1 and
decode batch 376. `INFERENCE_MAX_SEQ_LENGTH`, `INFERENCE_MAX_TOKENS`,
`INFERENCE_BUFFER_SIZE_GB`, `PREFILL_MAX_REQUESTS`, and
`DECODE_MAX_REQUESTS` are environment overrides. Ensure the sequence lengths,
precision, and speculative-decoding assumptions in the external benchmark
match the planner invocation before comparing its predicted latency or
throughput.

The performance server retains the smoke test's conservative cross-node
transport defaults (`UCX_TLS=cuda_copy,tcp,shm,cma,self`). Set
`UCX_TLS_OVERRIDE` and `UCX_MEMTYPE_CACHE_OVERRIDE` to the cluster's validated
RDMA/NIXL settings when measuring production transport performance.

The allocation launcher creates one two-node decode `srun`: task zero hosts the
Dynamo parent and rank-zero agent, while task one runs a headless rank agent.
The parent never creates a nested Slurm job step.

The smoke test defaults to `JOB_PORT_BASE=30000`, using ports
30000–30007 for the frontend, etcd, NATS, coordinator, and Dynamo status
endpoints. The 4P1D launcher defaults to `JOB_PORT_BASE=31000` and reserves
through `JOB_PORT_BASE+32`. Override `JOB_PORT_BASE` when running concurrent
jobs on nodes that share a network namespace. Mutable etcd and NATS JetStream
databases remain in the control node's container-local `/tmp` job directory.
Only logs and worker registration files are written to the shared worktree.

Override `LOAD_CHECKPOINT`, `TOKENIZER_MODEL`, `SERVED_MODEL_NAME`,
`CONTAINER_IMAGE`, `CONTAINER_MOUNTS`, ports, or the startup/request timeouts
through environment variables before submission. The four-GPU replica
parallelism is fixed to the validated TP=2, EP=4, expert-TP=1 profile. The
launcher leaves `CUDA_DEVICE_MAX_CONNECTIONS` unset for GB200; if this profile
is adapted to pre-Blackwell hardware, export `CUDA_DEVICE_MAX_CONNECTIONS=1`
before submission. The launcher uses the Nano disaggregation transport set
adapted to this fixed cross-node topology:
`UCX_TLS=cuda_copy,tcp,shm,cma,self` and `UCX_MEMTYPE_CACHE=n`. It excludes
`cuda_ipc`, which crashes when selected for a prefill-to-decode transfer across
these nodes, and `gdr_copy`, which cannot register the CUDA cache allocation.
Use `UCX_TLS_OVERRIDE` or `UCX_MEMTYPE_CACHE_OVERRIDE` to intentionally change
these settings.
