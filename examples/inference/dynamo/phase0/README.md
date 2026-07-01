# Megatron Aggregated Smoke Test

This single-node example launches NATS, etcd, one self-owned Megatron backend,
and the Dynamo frontend. The backend command is `python -m megatron.inference.integrations.dynamo`;
it starts its own local torchrun rank group and registers only after all ranks
have loaded the model.

## Launch

Run from an allocation with the Dynamo Megatron container available:

```bash
export DMG_SQSH=/path/to/dynamo-megatron.sqsh
export STAGE=/path/to/model/staging
bash examples/inference/dynamo/phase0/launch_phase0.sh
```

Useful overrides are `MODEL_CHECKPOINT`, `TOKENIZER_MODEL`,
`SERVED_MODEL_NAME`, `TP`, `HTTP_PORT`, and `COORD_PORT`. `TP` is also the
number of local GPUs assigned to the one engine replica.

The orchestrator prints `PHASE0_READY` after `/v1/models` exposes the model and
writes `/tmp/phase0.env` for the test scripts. Backend output, including model
load and streaming frames, is in `/tmp/worker.log`.

## Direct Command

The backend portion is equivalent to:

```bash
python -m megatron.inference.integrations.dynamo \
  --role aggregated \
  --model "$TOKENIZER_MODEL" \
  --served-model-name "$SERVED_MODEL_NAME" \
  --nproc-per-node "$TP" \
  --coordinator-host 127.0.0.1 \
  --coordinator-port "$COORD_PORT" \
  --megatron-root /opt/megatron-lm \
  -- \
  --tensor-model-parallel-size "$TP" \
  --pipeline-model-parallel-size 1 \
  --load "$MODEL_CHECKPOINT" \
  --tokenizer-type HuggingFaceTokenizer \
  --tokenizer-model "$TOKENIZER_MODEL" \
  <model architecture and dynamic-inference arguments>
```

No `--coordinator-addr`, Megatron HTTP frontend, or separately managed engine
process is involved.
