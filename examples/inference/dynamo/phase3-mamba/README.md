# Phase-3 hybrid disagg — Mamba state transfer (Nemotron-3 Nano)

Disaggregated prefill→decode for **hybrid (Mamba-2 + attention) models**. Hybrid
models keep two kinds of per-request state:

| state | where it lives | transferred by |
| --- | --- | --- |
| attention KV cache | block-indexed `memory_buffer` | `KvTransferAgent` (already) |
| Mamba conv/ssm state | block-indexed slot pool in `MambaSlotAllocator` | **new** conv/ssm `KvTransferAgent`s |

Before this change, only attention KV was handed off; the decode engine restored
Mamba state from its own (empty) cache, fell back to **zeroing** it, and decoded
hybrid models from zero recurrent state — silently wrong output. This example
transfers the Mamba conv/ssm state over the same NIXL path and verifies it.

## What changed (core)

`Megatron-LM/megatron/core/inference/engines/dynamic_engine.py`:

- `setup_kv_transfer` — for hybrid models, brings up two extra NIXL agents over
  the Mamba cache's `conv_states` / `ssm_states` slot pools (layout
  `[num_mamba_layers, max_slots, …]`, which `KvTransferAgent` already understands
  with the slot as the "block"). Guarded to **matched TP=1/PP=1** — conv/ssm
  state has no TP/PP reshard plan yet, so anything else fails loudly at launch.
- `_capture_handoff_meta` (prefill) — nests a `"mamba"` entry inside `kv_meta`
  listing, per handoff block with a committed snapshot, its `[position, slot]`.
  Rides the existing handoff transport verbatim (no coordinator/transport change).
- `add_request_with_kv_handoff` + `_import_mamba_handoff` (decode) — pulls the
  conv/ssm slots over NIXL into freshly allocated local slots, then registers
  them in `block_to_slot` + `hash_to_block_id` so the **existing** Mamba
  prefix-cache restore path takes over (`_find_mamba_match_count` →
  `_compute_prefix_match` sets `prefix_skip_tokens` to the matched Mamba block
  boundary → `restore_to_live`). The short tail past that boundary is
  re-prefilled, advancing the recurrence to the exact handoff position — no
  off-by-one, no zeroed state.

Emits `DISAGG_DECODE_MAMBA_IMPORT request_id=… mamba_blocks=…` on the decode
coordinator log.

## Run

From an `salloc`'d shell on the login node, use the wrapper — it srun's into the
container and binds the checkpoint paths for you:

```bash
export DMG_SQSH=/lustre/.../dynamo-megatron.sqsh   # container with nixl
export STAGE=/lustre/fsw/portfolios/nemotron/users/csathe
bash launch_mamba.sh           # prints PHASE3_MAMBA_READY
```

Or, if you're already inside the container with the checkpoint mounted:

```bash
# node with >=4 GPUs (prefill EP=2, decode EP=2)
bash orchestrate.sh            # prints PHASE3_MAMBA_READY
```

Run the tokenizer/model-card preflight before spending time loading the model:

```bash
PREFLIGHT_ONLY=1 bash orchestrate.sh
# PHASE3_MAMBA_PREFLIGHT_OK
```

The preflight resolves only the HF configuration/tokenizer files; it does not
download the HF weights. The actual weights still come exclusively from
`MODEL_CHECKPOINT`.

Then, in a second shell inside the same container:

```bash
source /tmp/phase3_mamba.env
bash verify_mamba.sh
```

### Mounting the checkpoint

The Nano artifacts live under `/lustre/fsw/portfolios/llmservice/...`, a
different portfolio than `$STAGE`, so a single `$STAGE` bind doesn't reach them.
`launch_mamba.sh` binds each artifact's directory into the container at the
**same absolute path** (read-only) via pyxis `--container-mounts`, so
orchestrate.sh's `--load` / `--pretrained-checkpoint` resolve unchanged. It
dedupes shared parent dirs (ckpt + pretrained under the same `users/` dir mount
once). Point at your own copy by exporting `MODEL_CHECKPOINT` /
`PRETRAINED_CHECKPOINT` / `DYNAMO_MODEL`, or add extra binds with
`EXTRA_MOUNTS="src:dst,src2:dst2"`.

`orchestrate.sh` defaults `MODEL_CHECKPOINT` and `PRETRAINED_CHECKPOINT` to the
cluster-staged Nemotron-3 Nano v3 artifacts (the same ones the Nano v3
functional test uses), and its `MODEL_ARGS` mirror the known working Nano
dynamic-serving config (`--model-provider hybrid`, EP sharding, chunked prefill,
flash attention, CUDA graph block scope, etc.). Override any of them to point at
your own copy. The checkpoint is consumed only by `orchestrate.sh` (via
`--load`); `verify_mamba.sh` just hits the running HTTP endpoint and needs
nothing checkpoint-related.

The checkpoint restores Megatron's native tiktoken vocabulary from an absolute
Lustre path. `TOKENIZER_MODEL` defaults to that same `*.vocab.json`, and the
launcher mounts its parent directory into the container before passing it as
`--tokenizer-model`. Override it when your checkpoint's vocabulary lives
elsewhere. Dynamo's worker registration is separate: use
`DYNAMO_MODEL` for the model directory / HF id with `config.json` and tokenizer
metadata. It defaults to `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16`. Do not
point it at the Megatron `nanov3` pretrained directory unless that directory
also contains HF-style `config.json` and `tokenizer.json`. The orchestrator
resolves an HF id with `ignore_weights=True` once, then passes that local
metadata directory to all Dynamo workers. This avoids downloading the 30B HF
weights during worker registration.

It brings up the disagg stack with fixed `TP=1`, `PP=1`, `EP=2` per role:
prefill uses `GPU_PREFILL=0,1` and decode uses `GPU_DECODE=2,3` by default.
`WITH_BASELINE=0` is the default because those four GPUs are already claimed by
the disagg roles. Set `WITH_BASELINE=1` only when you have two spare GPUs for an
aggregated `EP=2` reference, for example `GPU_BASELINE=4,5`.

`verify_mamba.sh` uses `/v1/completions` so both stacks receive the exact same
raw prompt without chat-template transformations. Megatron may use its native
checkpoint vocabulary internally, while Dynamo requires the equivalent HF
`tokenizer.json` to expose this HTTP endpoint.

EP is pinned to 2 by default (`ROLE_EP_SIZE=2`), with `TP=1` and `PP=1`. Mamba
handoff is gated on TP/PP only; EP shards experts while leaving the transferred
attention KV and Mamba conv/SSM state layout matched rank-to-rank.

## How verify proves correctness

`verify_mamba.sh`:

1. Confirms the decode log shows `DISAGG_DECODE_IMPORT` **and**
   `DISAGG_DECODE_MAMBA_IMPORT` with `mamba_blocks >= 1` (else the prompt was too
   short to commit a Mamba block boundary and the transfer path was untested →
   FAIL).
2. Greedy-decodes the same prompt through the disagg stack and the aggregated
   baseline and asserts the token text is **byte-identical**. Greedy decoding is
   deterministic and the handoff only moves bytes, so any divergence means the
   transferred state is wrong. An attention-only coherence check would pass even
   with zeroed Mamba state — this diff is what actually catches the bug.

## Key knobs

| env | default | meaning |
| --- | --- | --- |
| `MODEL_CHECKPOINT` | `…/ksanthanam/nemotron-3-nano-30b` | mcore checkpoint served via `--load` |
| `PRETRAINED_CHECKPOINT` | `…/ksanthanam/nanov3` | `--pretrained-checkpoint` source |
| `TOKENIZER_MODEL` | `…/multiMixV8….vocab.json` | native Megatron tokenizer path; its parent is mounted into the container |
| `DYNAMO_MODEL` | `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16` | model dir / HF id passed to Dynamo `--model` |
| `PREFLIGHT_ONLY` | `0` | set to `1` to resolve/validate tokenizer metadata and exit before GPU startup |
| `INFER_BUFFER_GB` | `20` | dynamic-batching KV buffer budget per engine |
| `INFER_MAX_TOKENS` | `8192` | dynamic batching token budget |
| `INFER_MAX_REQUESTS` | `256` | dynamic batching request cap |
| `MAMBA_GB` | `4.0` | Mamba state-cache budget (both prefill + decode) |
| `PREFIX_CACHE` | `1` | enable prefix caching (required for the handoff path) |
| `ROLE_EP_SIZE` | `2` | expert-model-parallel size per role; TP/PP stay 1 |
| `WITH_BASELINE` | `0` | also launch the aggregated reference for the token diff |
| `GPU_PREFILL`/`GPU_DECODE`/`GPU_BASELINE` | `0,1`/`2,3`/`4,5` | GPU assignment |
| `MODEL_ARGS_OVERRIDE` | _(unset)_ | replace the entire Nemotron arg block |

## Scope / limitations

- **Matched TP=1 / PP=1 only.** Mamba conv/ssm state is sharded across TP (head
  group) and PP (layer); unlike attention KV heads there is no reshard plan for
  it yet. Hybrid + (TP>1 or PP>1) with KV transfer raises `NotImplementedError`
  at launch rather than producing wrong output. Extending to heterogeneous
  TP/PP requires a conv/ssm reshard plan analogous to
  `disaggregation/kv_reshard.py`.
- Both prefill and decode must run with `--inference-dynamic-batching-prefix-caching`
  and `--inference-dynamic-batching-prefix-caching-mamba-gb`: prefill is the
  source of the block-boundary snapshots, decode restores them.
