# Nemotron Ultra native disaggregated deployment

This is a minimal two-node deployment with one OpenAI-compatible endpoint:

- node 0: four-GPU GB200 prefill shard, TP=4, EP=4, expert-TP=1, dense-DP=1;
- node 1: four-GPU GB200 decode shard, TP=4, EP=4, expert-TP=1, dense-DP=1;
- NIXL transfers KV and Mamba state from prefill to decode; and
- the native Megatron coordinator exposes the endpoint from the first node.

Submit from a shared Megatron worktree visible on both nodes:

```bash
JOB_ID=$(sbatch --parsable \
  --account=<account> \
  --partition=<partition> \
  examples/inference/nemotron_ultra_disagg/launch_slurm.sh)
echo "Submitted ${JOB_ID}"
```

To run in a Pyxis/Enroot container, pass the image in the submission
environment:

```bash
export CONTAINER_IMAGE=/path/to/megatron-nixl.sqsh
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
