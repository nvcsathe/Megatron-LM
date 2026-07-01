# Megatron Dynamo integration

Megatron-LM owns its Dynamo backend adapter, engine protocol, tests, examples,
deployment manifests, and runtime image. Dynamo remains an external dependency
that supplies the common backend API, distributed runtime, frontend, and KV
router.

Each registered worker owns one complete Megatron model replica. The lightweight
Dynamo parent launches a private TP/PP/EP rank group and Megatron coordinator,
then connects to it through `InferenceClient`. Individual model-parallel ranks
are not registered as Dynamo workers.

## Layout

```text
megatron/inference/integrations/dynamo/   backend adapter and engine protocol
megatron/core/inference/disaggregation/  engine-native KV/state handoff
tests/unit_tests/inference/dynamo/        adapter unit tests
examples/inference/dynamo/                Slurm test
integrations/dynamo/container/            Megatron-owned runtime image
integrations/dynamo/deploy/               DynamoGraphDeployment manifests
```

Logic reusable by other inference engines belongs in Dynamo's
`dynamo.common.backend`; Megatron-specific behavior belongs here.

## Build the image

Build from the Megatron-LM repository root. The default Dynamo ref is recorded
in `integrations/dynamo/dynamo-ref.txt`; use a release, branch, tag, or commit to
keep the adapter and common backend API compatible.

```bash
export IMAGE=megatron-dynamo:dev
export DYNAMO_REF="$(cat integrations/dynamo/dynamo-ref.txt)"

docker build \
  -f integrations/dynamo/container/Dockerfile \
  --build-arg DYNAMO_REF="$DYNAMO_REF" \
  -t "$IMAGE" \
  .
```

For a Dynamo fork, also pass `--build-arg DYNAMO_REPO=<git-url>`. Numeric refs
install the matching PyPI release; other non-empty refs install Dynamo and its
native runtime from git.

The image contains Megatron-LM, the adapter, `ai-dynamo`,
`ai-dynamo-runtime`, NIXL, and the dependencies needed by the inference path.
It also includes NATS and etcd binaries for the single-container examples;
production deployments normally run those as separate services.

Validate the ownership boundary:

```bash
docker run --rm "$IMAGE" python -c '
from megatron.inference.integrations.dynamo.llm_engine import MegatronLLMEngine
from megatron.inference.integrations.dynamo.engine_service import main
from dynamo.common.backend import LLMEngine
assert issubclass(MegatronLLMEngine, LLMEngine)
print("Megatron adapter and Dynamo runtime are present")
'
```

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
  --kv-transfer-listen-addr 10.0.0.12:7000 \
  -- <Megatron arguments>

python -m megatron.inference.integrations.dynamo \
  --role decode --component backend \
  --model Qwen/Qwen3-8B --nproc-per-node 4 \
  --coordinator-host 10.0.0.13 \
  --kv-transfer-listen-addr 10.0.0.13:7000 \
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

The Nano v3 Slurm test starts etcd, NATS, the frontend, and matched TP=1/PP=1
prefill and decode workers:

```bash
bash examples/inference/dynamo/nano-v3-test/launch.sh
```

## Tests

Adapter tests require an environment containing both Megatron and Dynamo:

```bash
pytest -q tests/unit_tests/inference/dynamo
pytest -q tests/unit_tests/inference/test_dynamo_engine_service.py
pytest -q tests/unit_tests/inference/test_kv_transfer_backends.py
```

## Runtime contract

- Prefill runs a zero-token request, pins prompt blocks, and returns NIXL
  metadata in `disaggregated_params`.
- The frontend forwards the prefill result to the selected decode worker.
- Decode imports the blocks before generation and releases the source handoff
  after the first post-import output.
- Prefix block events cross the private coordinator and are published by the
  parent as logical DP rank zero.
- Cancellation targets the exact Megatron request; shutdown unregisters the
  endpoint, drains requests and pinned handoffs, and then stops all ranks.

The current launcher supports one node per engine. Scale horizontally by
adding complete Dynamo component replicas.
