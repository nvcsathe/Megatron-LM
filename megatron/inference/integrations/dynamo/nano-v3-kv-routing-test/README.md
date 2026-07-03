# Nano v3 KV-event and routing tests

This Slurm harness validates the Dynamo-Megatron KV event and aggregated
KV-routing path for Nemotron-3 Nano. It runs one complete Megatron replica per
Dynamo worker. Model-parallel ranks remain private to that worker.

The default two-node topology is:

```text
2 nodes x 4 GPUs
2 aggregated workers per node
2 GPUs per worker: TP=1, PP=1, EP=2
4 independent Dynamo workers and KV caches
```

Set `ROLE_EP_SIZE=4` to run one EP=4 worker per node instead. EP=2 is the
recommended routing and baseline topology because it provides four routing
targets. EP=4 is useful as an additional event-deduplication smoke test.

## Prerequisites

- An active Slurm allocation containing all test nodes and GPUs.
- A shared `STAGE` directory visible at the same path on every node.
- A Dynamo-Megatron squashfs image in `DMG_SQSH`.
- The Nano checkpoint, pretrained checkpoint, and tokenizer used by
  `../nano-v3-test`, or overrides for their environment variables.

The tests are intentionally sequential. Each launch starts fresh workers, so a
previous test's physical KV cache cannot affect the next test.

## Run the tests

From an allocation containing at least two 4-GPU nodes (the test selects two
nodes and uses eight GPUs total):

```bash
export DMG_SQSH=/path/to/dynamo-megatron.sqsh
export STAGE=/path/to/shared/staging

bash megatron/inference/integrations/dynamo/nano-v3-kv-routing-test/launch-events.sh
bash megatron/inference/integrations/dynamo/nano-v3-kv-routing-test/launch-routing.sh
bash megatron/inference/integrations/dynamo/nano-v3-kv-routing-test/launch-baseline.sh
```

The routing launch creates `DATASET_PATH`; the baseline launch requires and
reuses that exact JSONL file. By default it is:

```text
$STAGE/nano-v3-kv-routing/dataset.jsonl
```

Every launch writes logs, request records, and a summary under:

```text
$STAGE/nano-v3-kv-routing/runs/$SLURM_JOB_ID-$TEST_MODE/
```

Use a unique `RUN_ID` when repeating a mode inside the same allocation.

## What each test checks

### `launch-events.sh`

- Starts the frontend with event-driven KV routing. Approximate routing is not
  enabled.
- Loads every logical Dynamo worker ID from the adapter's structured readiness records.
- Directs a distinct long prompt to every worker and repeats it.
- Verifies that the response identifies the requested worker and logical DP
  rank zero.
- Requires exactly one multi-block `stored` event per logical worker, which
  detects duplicate publication by EP ranks, and rejects invalid,
  missing-parent, or missing-block indexer statuses.
- Requires no `removed` or `cleared` events between the initial request and its
  replay. Idle EP dummy forwards preserve the KV and Mamba prefix caches.
- Requires every worker log to gain exactly one Megatron prefix-cache hit from
  the repeated request.

The event prompt is kept below the default prefill window so its many blocks
are carried in one event message. Do not reduce `INFER_MAX_TOKENS` below that
prompt size when running this exact-count assertion.

All three modes require the `lru` prefix-cache eviction policy so completed
requests remain available to later requests. The launcher rejects `ref_zero`,
which deliberately removes blocks as soon as their last active reference is
released and therefore cannot validate sequential cache reuse.

### `launch-routing.sh`

- Regenerates the deterministic shared-prefix dataset.
- Warms one request from each prefix family.
- Waits for KV events to settle, then replays later turns concurrently.
- Records Dynamo worker IDs and router timing metadata for every request.
- Requires the configured minimum affinity between each prefix family and its
  warm-up worker.
- Reports both Dynamo's predicted hit rate and Megatron's actual cumulative
  prefix-cache block matches.

### `launch-baseline.sh`

- Starts the same number of workers with identical model and cache settings.
- Changes only the frontend policy to round-robin.
- Replays the exact routing dataset without regenerating it.
- Reports Megatron's actual cache-hit counters for comparison with the routing
  summary.

The baseline is aggregated and round-robin. It is not a disaggregated
prefill/decode architecture comparison.

## Main settings

| Variable | Default | Purpose |
| --- | --- | --- |
| `TEST_NNODES` | `2` | Slurm nodes selected from the allocation |
| `ROLE_EP_SIZE` | `2` | GPUs and expert-parallel ranks per worker |
| `GPUS_PER_NODE` | `4` | GPUs visible to each Slurm node task |
| `NAMESPACE` | `nano-kv-test` | Dynamo namespace |
| `CONTEXT_LENGTH` | `8192` | Served context length |
| `KV_BLOCK_SIZE` | `256` | Megatron dynamic-batching block size |
| `INFER_BUFFER_GB` | `20` | Inference buffer per worker |
| `INFER_MAX_REQUESTS` | `16` | Request slots; bounds Mamba extraction scratch |
| `MAMBA_GB` | `4.0` | Mamba prefix-cache budget |
| `PREFIX_CACHE_EVICTION_POLICY` | `lru` | Retain completed-request blocks for sequential reuse |
| `CUDA_GRAPH_IMPL` | `none` | Set to `local` to opt into CUDA-graph warmup |
| `PREFIX_FAMILIES` | `32` | Shared-prefix families in the routing dataset |
| `TURNS_PER_FAMILY` | `8` | Growing requests per family |
| `PREFIX_REPEAT` | `3072` | Repeated words in each shared prefix |
| `ROUTING_CONCURRENCY` | `4` | Concurrent measured routing requests |
| `BASELINE_CONCURRENCY` | `4` | Concurrent baseline requests |
| `MAX_TOKENS` | `32` | Output tokens per workload request |
| `MIN_AFFINITY` | `0.95` | Minimum post-warmup family affinity |
| `FRONTEND_SETTLE_SECONDS` | `5` | Registration propagation delay before the driver starts |
| `EVENT_SETTLE_SECONDS` | `5` | Delay after warm-up before measured routing |
| `TURN_SETTLE_SECONDS` | `1` | Delay between later conversation turns |
| `LOG_SETTLE_SECONDS` | `2` | Delay before cumulative counters are parsed |
| `DATASET_PATH` | shared path above | Dataset created by routing and consumed by baseline |
| `MODEL_ARGS_OVERRIDE` | unset | Replace the default Nano Megatron argument list |

Use `TEST_TIMEOUT_SECONDS` to bound the Python test driver and
`WORKER_START_TIMEOUT` to change the model startup timeout.

## Comparing routing and baseline

The summaries expose two actual Megatron metrics:

```text
request_hit_rate = cumulative prefix_cache_hits / completed requests
block_hit_rate   = cumulative prefix_cache_blocks_matched
                   / sum(floor(prompt_tokens / KV_BLOCK_SIZE))
```

Counters are read once per logical worker log using the largest cumulative
value. They are not summed across EP ranks or across cumulative log lines.
Dynamo's `nvext.timing.kv_hit_rate` is retained as a predicted routing metric,
not used as ground truth.

Compare two completed summaries with:

```bash
python megatron/inference/integrations/dynamo/nano-v3-kv-routing-test/compare.py \
  "$STAGE/nano-v3-kv-routing/runs/<routing-run>/summary.json" \
  "$STAGE/nano-v3-kv-routing/runs/<baseline-run>/summary.json"
```
