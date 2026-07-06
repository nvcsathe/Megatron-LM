# Nano v3 2P2D disaggregation test

This Slurm harness runs Nemotron-3 Nano disaggregated inference on exactly
eight GPUs:

```text
2 prefill workers x EP=2 = 4 GPUs
2 decode workers  x EP=2 = 4 GPUs
TP=1, PP=1
```

By default the workers are placed on two four-GPU nodes. Node zero owns both
prefill workers and node one owns both decode workers. Set `TEST_NNODES=1` and
`GPUS_PER_NODE=8` to put the same topology on one eight-GPU node.

The Nano model, cache, and Mamba settings match `../nano-v3-test` and
`../nano-v3-kv-routing-test`. Each worker is a separate Dynamo replica with a
private Megatron rank group and cache.

## Run

From an allocation with either two four-GPU nodes or one eight-GPU node:

```bash
export DMG_SQSH=/path/to/dynamo-megatron.sqsh
export STAGE=/path/to/shared/staging
bash megatron/inference/integrations/dynamo/nano-v3-2p2d-test/launch.sh
```

The launcher selects eight GPUs, starts NATS and etcd on the first node,
launches all four workers, starts the KV-aware frontend, and runs the workload.
Logs and the JSON summary are written under:

```text
$STAGE/nano-v3-2p2d/runs/$SLURM_JOB_ID/
```

Use a distinct `RUN_ID` to repeat the test inside one allocation.

## Assertions

The workload requires:

- exactly two registered prefill and two registered decode worker identities;
- successful, non-empty completions that collectively use every worker;
- both decode workers to log attention-KV import; and
- both decode workers to import at least one committed Mamba state block.

Requests are issued in concurrent rounds until all workers have been observed
or `MAX_REQUESTS` is reached.

## Main settings

| Variable | Default | Purpose |
| --- | --- | --- |
| `TEST_NNODES` | `2` | nodes selected from the allocation; must be 1 or 2 |
| `GPUS_PER_NODE` | `4` | GPUs used per selected node; use 8 with one node |
| `ROLE_EP_SIZE` | `2` | GPUs and expert-parallel ranks per worker |
| `CONTEXT_LENGTH` | `8192` | served context length |
| `KV_BLOCK_SIZE` | `256` | dynamic-batching cache block size |
| `INFER_BUFFER_GB` | `20` | inference buffer per worker |
| `INFER_MAX_REQUESTS` | `16` | request slots per worker |
| `MAMBA_GB` | `4.0` | Mamba prefix/state-cache budget |
| `REQUEST_CONCURRENCY` | `4` | requests issued per workload round |
| `MAX_REQUESTS` | `32` | maximum requests used to cover all workers |
| `MODEL_ARGS_OVERRIDE` | unset | replace the default Nano Megatron arguments |

