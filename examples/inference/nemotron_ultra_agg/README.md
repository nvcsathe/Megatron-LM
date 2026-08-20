# Nemotron Ultra aggregated deployment

This launcher translates NVIDIA's
[official Nemotron 3 Ultra BF16 deployment profile](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16#vllm)
to Megatron inference on two four-GPU GB200 nodes. It starts one aggregated
prefill/decode model replica and exposes an OpenAI-compatible endpoint.

The default model and runtime settings are:

- eight GB200 GPUs total: TP=8, EP=8, expert-TP=1, PP=1, and dense-DP=1;
- BF16 weights with the official fast Hugging Face tokenizer and chat template;
- 262,144-token maximum context, 32,768 batched tokens, and 16 requests;
- chunked prefill and five-token MTP speculative decoding;
- sigmoid MoE routing, grouped GEMM, all-to-all dispatch, and shared-expert overlap;
- the inference-optimized transformer and vLLM grouped-GEMM backend; and
- the Nemotron v3 reasoning parser and Qwen3 Coder tool parser.

These settings correspond to the official Ultra profile where Megatron has an
equivalent option. The 25-GiB dynamic-batching buffer is the Megatron equivalent
of reserving inference cache capacity. Prefix caching is opt-in because
Megatron's hybrid Mamba prefix cache requires a separate fixed HBM budget.
The vLLM-specific Ray executor, cache stochastic rounding, FlashInfer Mamba
backend, and multithreaded Safetensors loader have no direct Megatron inference
CLI equivalents and are therefore not passed through.

Submit from the shared worktree:

```bash
cd /lustre/fsw/portfolios/nemotron/users/csathe/Megatron-LM
export MEGATRON_ROOT="$PWD"
JOB_ID=$(sbatch --parsable --export=ALL \
  --account=<account> \
  --partition=<partition> \
  examples/inference/nemotron_ultra_agg/launch_slurm.sh)
echo "Submitted ${JOB_ID}"
```

The launcher uses the same Dynamo/Megatron container as the Nano test by
default and installs Quart/Hypercorn into the shared runtime overlay if needed.
It writes the endpoint to `ultra-agg-${SLURM_JOB_ID}.endpoint`.

Send a completion request:

```bash
BASE_URL=$(cat "ultra-agg-${JOB_ID}.endpoint")
curl "${BASE_URL}/completions" \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "nemotron-ultra",
    "prompt": "Explain aggregated inference in one sentence.",
    "max_tokens": 64,
    "temperature": 0
  }'
```

For an OpenAI chat request, message `content` must currently be a string rather
than a multimodal content-part list:

```bash
curl "${BASE_URL}/chat/completions" \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "nemotron-ultra",
    "messages": [{"role": "user", "content": "What is 2+2?"}],
    "max_tokens": 64,
    "temperature": 0
  }'
```

If the official profile does not leave enough HBM with this Megatron
checkpoint, retain the 256K context but start with a smaller cache profile:

```bash
export INFERENCE_BUFFER_SIZE_GB=8
export INFERENCE_MAX_TOKENS=4096
export INFERENCE_MAX_REQUESTS=4
```

Prefix caching can be enabled after the baseline is stable:

```bash
export ENABLE_PREFIX_CACHING=1
export MAMBA_PREFIX_CACHE_GB=4
```

Checkpoint paths, container settings, ports, limits, and endpoint hostname are
all overrideable through the corresponding environment variables in the
launcher.
