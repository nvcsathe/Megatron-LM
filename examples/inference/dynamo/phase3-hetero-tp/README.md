# Phase-3 heterogeneous TP (prefill_TP != decode_TP)

Disaggregated prefill/decode where the two engines run at **different tensor-
parallel sizes**. The KV handoff re-shards KV heads across the TP boundary
instead of requiring a 1:1 rank match.

## What changed (vs phase3/)

Megatron core (`megatron/core/inference/disaggregation/`):
- `transfer_backends/nixl.py`
  - `export_meta()` now also emits TP topology (`tp_size`, `tp_rank`,
    `num_kv_heads_global`, `heads_per_partition`, `head_dim`,
    `tokens_per_block`).
  - `pull_blocks()` accepts a **single** peer meta (matched layout → original
    whole-slice copy) **or a list** of per-rank metas → head-aware re-shard.
  - The backend issues one NIXL transfer per contributing prefill rank;
    a head sub-range is strided across the `T` tokens of each `[T, H, d]` slice,
    so it emits one descriptor per `(block, outer, token)`.
- `kv_reshard.py`
  - Computes, per decode rank, which `(prefill_rank, head_range)` fragments to
    pull by intersecting global KV-head ranges.
- `inference_state_handoff.py`
  - `setup_kv_transfer()` passes TP topology into `make_agent` and
    `all_gather_object`s every prefill rank's meta once at startup.
  - `_capture_handoff_meta()` ships that gathered list (TP>1) so a different-TP
    decode peer can re-shard. The Dynamo workers/coordinator carry `kv_meta`
    opaquely, so no wire-protocol change was needed.

## Constraints

- One of `{TP_PREFILL, TP_DECODE}` must divide the other.
- Both must divide `num_query_groups` (KV heads). For Llama-3.1-8B that is 8,
  so valid TP values are `{1, 2, 4, 8}`. Past 8, KV heads replicate rather than
  partition — re-shard raises `NotImplementedError`; use matched TP there.
- PP and dtype must still match (only TP may differ). `num_outer` equality is
  asserted in `reshard_plan`. PP-heterogeneous is future work.

## Run (inside the combined container, on a node with >= TP_PREFILL+TP_DECODE GPUs)

```bash
# Example: prefill at TP=2, decode at TP=4 → needs 6 GPUs
TP_PREFILL=2 TP_DECODE=4 \
  MODEL_CHECKPOINT=/path/to/llama3.1-8b-mcore \
  ./orchestrate.sh        # blocks; prints PHASE3_HETERO_READY when up

# in a second shell in the same container:
source /tmp/phase3_hetero.env
./verify_hetero_tp.sh
```

Try the merge direction too: `TP_PREFILL=4 TP_DECODE=2`.

## Correctness testing

1. **Planner math (no GPU/NIXL)** — pure unit test of the head-range
   arithmetic, covers equal / split / merge / incompatible / replication:
   ```
   pytest tests/unit_tests/inference/test_kv_transfer_reshard.py   # in Megatron-LM
   ```
   (Run inside the container per the build-and-dependency skill — not on the host.)

2. **End-to-end** — `verify_hetero_tp.sh` confirms greedy output is coherent
   and the decode engine imported (did not re-prefill).

3. **Gold standard (manual)** — run the same greedy (`temperature=0`) prompt
   through a matched-TP stack (`../phase3/`) and diff the token sequences. They
   must be identical: re-shard only relocates bytes, it does not change the
   math. Any divergence means a head-mapping or offset bug.
