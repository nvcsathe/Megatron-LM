# Nano v3 Mamba disaggregation test

This Slurm test launches matched TP=1/PP=1 prefill and decode workers for
Nemotron-3 Nano, transfers attention KV plus Mamba conv/SSM state over NIXL,
and verifies that decode imported both kinds of state.

## Run

From an `salloc` shell:

```bash
export DMG_SQSH=/path/to/dynamo-megatron.sqsh
export STAGE=/path/to/staging
bash megatron/inference/integrations/dynamo/nano-v3-test/launch.sh
```

The launcher defaults to the cluster-staged Nano v3 checkpoint, pretrained
checkpoint, and tokenizer. Override `MODEL_CHECKPOINT`,
`PRETRAINED_CHECKPOINT`, `TOKENIZER_MODEL`, or `DYNAMO_MODEL` when using other
artifacts. Use `EXTRA_MOUNTS=src:dst,...` for additional container mounts.

To validate model metadata without loading the model:

```bash
PREFLIGHT_ONLY=1 bash megatron/inference/integrations/dynamo/nano-v3-test/launch.sh
```

After the stack reports `NANO_V3_TEST_READY`, run in a second shell attached to
the same container:

```bash
source /tmp/nano_v3_test.env
bash /opt/megatron-lm/megatron/inference/integrations/dynamo/nano-v3-test/verify.sh
```

The verifier requires a non-empty completion, a KV import marker, and a Mamba
import with at least one committed block. With `WITH_BASELINE=1`, it also
requires byte-identical greedy output from an aggregated reference worker.
The baseline needs two additional GPUs by default.

## Main settings

| Variable | Default | Purpose |
| --- | --- | --- |
| `ROLE_EP_SIZE` | `2` | expert parallel size per worker |
| `GPU_PREFILL` / `GPU_DECODE` | `0,1` / `2,3` | disaggregated worker GPUs |
| `WITH_BASELINE` | `0` | enable exact output comparison |
| `GPU_BASELINE` | `4,5` | baseline worker GPUs |
| `CONTEXT_LENGTH` | `4096` | served context length |
| `INFER_BUFFER_GB` | `20` | KV buffer per worker |
| `MAMBA_GB` | `4.0` | Mamba state-cache budget |
| `MODEL_ARGS_OVERRIDE` | unset | replace the Nano model argument list |

Both prefill and decode require prefix caching and the Mamba state cache.
Mamba handoff currently requires matched TP=1/PP=1; EP may be greater than one.
