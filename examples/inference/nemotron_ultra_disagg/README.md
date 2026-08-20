# Nemotron Ultra native disaggregated deployment

This is a four-node BF16 deployment with one OpenAI-compatible endpoint:

- nodes 0–1: eight-GPU GB200 prefill shard, TP=8, EP=8, expert-TP=1, dense-DP=1;
- nodes 2–3: eight-GPU GB200 decode shard, TP=8, EP=8, expert-TP=1, dense-DP=1;
- NIXL transfers KV and Mamba state from prefill to decode; and
- the native Megatron coordinator exposes the endpoint from the first node.

The launcher follows the Nano-v3 evaluation path: it passes the Dynamo/NIXL
`.sqsh` directly to `srun`, mounts `/home` and `/lustre`, sets the shared
Megatron worktree as the container workdir, and launches the container's plain
`python`. It does not invoke `uv`, so the image's `/opt/dynamo/venv` remains the
active Python environment.

Before starting the distributed workers, the launcher opens a one-node
container step and installs `quart==0.20.0` (including Hypercorn) into the
shared `${MEGATRON_ROOT}/.runtime-deps/nemotron-ultra-py312` overlay when it is
not already importable. The overlay is added to `PYTHONPATH` for every worker
and reused by later jobs. Override its location with `PYTHON_DEPS_DIR`.

Submit from the root of a shared Megatron worktree visible on all four nodes.
The launcher uses `SLURM_SUBMIT_DIR` as `MEGATRON_ROOT`, avoiding SLURM's
node-local spool copy of the batch script:

```bash
cd /lustre/fsw/portfolios/nemotron/users/csathe/Megatron-LM
export MEGATRON_ROOT="$PWD"
JOB_ID=$(sbatch --parsable \
  --account=<account> \
  --partition=<partition> \
  examples/inference/nemotron_ultra_disagg/launch_slurm.sh)
echo "Submitted ${JOB_ID}"
```

By default this uses the same image as the Nano-v3 test:

```text
/lustre/fsw/portfolios/nemotron/users/csathe/chaitrasathe+dynamo-megatron+mamba.sqsh
```

Override it when needed by exporting `CONTAINER_IMAGE` before submission:

```bash
export CONTAINER_IMAGE=/path/to/megatron-nixl.sqsh
export MEGATRON_ROOT="$PWD"
JOB_ID=$(sbatch --parsable --export=ALL \
  --account=<account> \
  --partition=<partition> \
  examples/inference/nemotron_ultra_disagg/launch_slurm.sh)
```

The job prints the endpoint and writes it to
`ultra-disagg-${SLURM_JOB_ID}.endpoint` in the worktree. Send an OpenAI
completion request with:

```bash
BASE_URL=$(cat "ultra-disagg-${JOB_ID}.endpoint")
curl "${BASE_URL}/completions" \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "nemotron-ultra",
    "prompt": "Explain disaggregated inference in one sentence.",
    "max_tokens": 64,
    "temperature": 0
  }'
```

The checkpoint paths, ports, cache sizes, endpoint hostname, and container
settings can all be overridden through their corresponding environment
variables in `launch_slurm.sh`. `ENDPOINT_HOST` is useful when the first
node's scheduler hostname is not directly reachable by clients.
