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

The batch allocation derives a unique block of frontend, etcd, NATS,
coordinator, and Dynamo status ports from `SLURM_JOB_ID`. It also keeps the
mutable etcd and NATS JetStream databases in the control node's container-local
`/tmp/nemotron-super-dynamo-disagg-${SLURM_JOB_ID}`. Only logs and worker
registration files are written to the shared worktree.

Override `LOAD_CHECKPOINT`, `TOKENIZER_MODEL`, `SERVED_MODEL_NAME`,
`CONTAINER_IMAGE`, `CONTAINER_MOUNTS`, ports, or the startup/request timeouts
through environment variables before submission. The four-GPU replica
parallelism is fixed to the validated TP=2, EP=4, expert-TP=1 profile. The
launcher leaves `CUDA_DEVICE_MAX_CONNECTIONS` unset for GB200; if this profile
is adapted to pre-Blackwell hardware, export `CUDA_DEVICE_MAX_CONNECTIONS=1`
before submission. The launcher defaults `UCX_TLS` to `cuda_copy,tcp`, the
portable cross-node NIXL/UCX path used by this smoke test. Override `UCX_TLS`
and set `UCX_NET_DEVICES` before submission to test a configured RDMA fabric.
The default excludes `cuda_ipc`, which is valid only within a host, and
`gdr_copy`, which cannot register these stream-ordered/expandable CUDA
allocations.
