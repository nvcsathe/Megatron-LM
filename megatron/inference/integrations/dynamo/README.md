# Megatron Dynamo integration

Megatron-LM owns its Dynamo backend adapter, engine protocol, and tests. Dynamo
remains an external dependency that supplies the common backend API,
distributed runtime, frontend, and KV router.

Each registered worker owns one complete Megatron model replica. The lightweight
Dynamo parent launches a private TP/PP/EP rank group and Megatron coordinator,
then connects to it through `InferenceClient`. Individual model-parallel ranks
are not registered as Dynamo workers.

## Layout

```text
megatron/inference/integrations/dynamo/             adapter and engine protocol
megatron/core/inference/engine_endpoint.py           shared endpoint contract
megatron/core/inference/disaggregation/            reusable KV/state handoff
tests/unit_tests/inference/dynamo/                  adapter unit tests
```

Logic reusable by other inference engines belongs in Dynamo's
`dynamo.common.backend`; Megatron-specific behavior belongs here.

## Launch

Pass integration arguments before `--` and normal Megatron arguments after it:

```bash
python -m megatron.inference.integrations.dynamo \
  --role aggregated \
  --model Qwen/Qwen3-8B \
  --served-model-name Qwen/Qwen3-8B \
  --nproc-per-node 4 \
  --megatron-root /opt/megatron-lm \
  -- \
  --load /models/qwen3-8b-megatron \
  --tensor-model-parallel-size 4 \
  --tokenizer-type HuggingFaceTokenizer \
  --tokenizer-model Qwen/Qwen3-8B \
  --inference-dynamic-batching \
  --inference-dynamic-batching-prefix-caching
```

Disaggregated serving requires separate prefill and decode workers. Each worker
starts its own private coordinator and rank group:

```bash
python -m megatron.inference.integrations.dynamo \
  --role prefill --component prefill \
  --model Qwen/Qwen3-8B --nproc-per-node 4 \
  --coordinator-host 10.0.0.12 \
  -- <Megatron arguments>

python -m megatron.inference.integrations.dynamo \
  --role decode --component backend \
  --model Qwen/Qwen3-8B --nproc-per-node 4 \
  --coordinator-host 10.0.0.13 \
  -- <Megatron arguments>
```

The frontend owns the `PrefillRouter` and embedded KV router. Enable KV-aware
routing explicitly:

```bash
python -m dynamo.frontend \
  --router-mode kv \
  --request-plane nats \
  --event-plane nats
```

The parent event socket binds to `127.0.0.1` by default. Only the local global
rank-zero process connects to this endpoint, so loopback also works for a
multi-node replica.

### Multi-node SLURM replica

The default `local` launcher starts one complete replica on one node. For a
multi-node replica, SLURM must create one task per node in a single job step.
Task zero runs the Dynamo parent and its local rank agent; every other task runs
the headless entrypoint, which starts only its local agent and never registers
a Dynamo worker. The parent does not invoke `srun`.

```bash
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
export MASTER_PORT=29500

# Run this script through: srun --nodes=$SLURM_NNODES --ntasks=$SLURM_NNODES \
#   --ntasks-per-node=1 bash launch-replica-node.sh
common_args=(
  --launcher external --nnodes "$SLURM_NNODES" --node-rank "$SLURM_PROCID"
  --nproc-per-node 8 --master-addr "$MASTER_ADDR" --master-port "$MASTER_PORT"
  --role aggregated --model Qwen/Qwen3-8B
)
engine_args=(--load /path/to/checkpoint) # Add the remaining Megatron arguments.
if [[ "$SLURM_PROCID" -eq 0 ]]; then
  exec python -m megatron.inference.integrations.dynamo \
    "${common_args[@]}" -- "${engine_args[@]}"
else
  exec python -m megatron.inference.integrations.dynamo.headless \
    "${common_args[@]}" -- "${engine_args[@]}"
fi
```

`--launcher external` is scheduler-neutral: the allocation layer supplies the
node rank and owns placement and hard cancellation. Reserve the selected nodes
for this complete replica; individual Megatron ranks are not Dynamo workers.

## Tests

Adapter tests require an environment containing both Megatron and Dynamo:

```bash
pytest -q tests/unit_tests/inference/dynamo
pytest -q tests/unit_tests/inference/test_engine_endpoint.py
pytest -q tests/unit_tests/inference/test_kv_transfer_backends.py
```

## Runtime contract

- The parent binds an engine-event socket before launch and passes its address
  to the child. Rank zero sends readiness, the request-coordinator address, and
  static engine capabilities as the first message on that socket.
- Normal requests, streaming replies, cancellation, and KV handoff commands use
  the ordinary Megatron `InferenceClient` protocol; the coordinator has no
  Dynamo mixins or Dynamo-only management headers.
- Prefill runs a zero-token request, pins prompt blocks, and returns NIXL
  metadata in `disaggregated_params`.
- The frontend forwards the prefill result to the selected decode worker.
- Decode imports the blocks before generation and releases the source handoff
  after the first post-import output.
- Rank zero queues prefix block events after successful forwards; a dedicated
  thread sends them directly to the Dynamo parent without crossing the request
  coordinator or stalling the forward path.
- Cancellation targets the exact Megatron request; shutdown unregisters the
  endpoint, drains active requests, and then stops all ranks.

The default launcher supports one node per engine. The external launcher can
join node-local agents into one multi-node engine; scale horizontally by adding
complete Dynamo component replicas on separate node sets.
