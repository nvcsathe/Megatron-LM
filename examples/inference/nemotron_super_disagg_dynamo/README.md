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
export MEGATRON_ROOT="$PWD"
export CONTAINER_IMAGE=/path/to/dynamo-megatron.sqsh

sbatch --export=ALL --account=<account> --partition=<partition> \
  examples/inference/nemotron_super_disagg_dynamo/run_slurm_test.sh
```

The default checkpoint and tokenizer are:

```text
/lustre/fsw/portfolios/nemotron/projects/nemotron_sw_pre/users/ksanthanam/nemotron-3-super-120b-a12b
nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16
```

The launcher starts the Dynamo control plane and both workers, waits for their
registrations, sends one deterministic `/v1/completions` request, validates
that the response contains generated text, and shuts everything down. Logs and
the response are retained under
`logs/nemotron-super-dynamo-disagg-${SLURM_JOB_ID}` for diagnosis.

Override `LOAD_CHECKPOINT`, `TOKENIZER_MODEL`, `SERVED_MODEL_NAME`,
`CONTAINER_MOUNTS`, ports, or the startup/request timeouts through environment
variables before submission. The four-GPU replica parallelism is fixed to the
validated TP=2, EP=4, expert-TP=1 profile. The launcher leaves
`CUDA_DEVICE_MAX_CONNECTIONS` unset for GB200; if this profile is adapted to
pre-Blackwell hardware, export `CUDA_DEVICE_MAX_CONNECTIONS=1` before
submission.
