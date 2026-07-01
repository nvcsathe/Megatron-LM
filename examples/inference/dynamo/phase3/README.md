# Phase-3 — Minimal disaggregated serving end-to-end

This directory mirrors `phase0/`, but brings up the full disagg topology and
adds the one assertion that actually proves disagg is working: greppable
markers in the prefill + decode engine logs.

## Building the container

Phase 3 needs NIXL in the image. Build the Megatron-owned backend image before
the first run:

```bash
# From the Megatron-LM repository root.
cd /path/to/Megatron-LM

# Build this checkout with the pinned Dynamo dependency.
docker build \
    -f integrations/dynamo/container/Dockerfile \
    --build-arg DYNAMO_REF="$(cat integrations/dynamo/dynamo-ref.txt)" \
    -t megatron-dynamo:phase3 \
    .

# Convert to sqsh for pyxis on the cluster.
enroot import -o dynamo-megatron+phase3-arm64.sqsh \
    dockerd://megatron-dynamo:phase3

# Push to lustre and update DMG_SQSH in your launch invocation.
rsync -av --progress dynamo-megatron+phase3-arm64.sqsh \
    csathe@cluster:/lustre/.../csathe/
```

If you only edited the Megatron *Python* code (not the requirements or the
Dockerfile), you do **not** need to rebuild — bind-mount the host checkout
via `MEGATRON_LOCAL_DEV` (see [Run](#run)).

### Verifying NIXL landed in the image

After the build but before pushing to the cluster:

```bash
docker run --rm megatron-dynamo:phase3 python -c \
    "from nixl._api import nixl_agent; print('nixl OK')"
```

## Topology

```
                              ┌── Megatron PREFILL engine (TP=N, GPUs 0..N-1)
                              │     - role=prefill, NIXL :7000
                              │
nats + etcd ── dynamo.frontend
                              │
                              └── Megatron DECODE engine  (TP=N, GPUs N..2N-1)
                                    - role=decode, NIXL :7001
                                    - prefix caching ENABLED
                                      (via --inference-dynamic-batching-prefix-caching;
                                       required for the prefill-skip path)
```

A request flows: frontend → `PrefillRouter` → prefill endpoint → prefill engine
runs prefill only, pins KV blocks, returns `disaggregated_params` → frontend
forwards `prefill_result` → decode endpoint → decode engine NIXL-pulls the KV
blocks, registers their hashes, then `add_request` finds them as
prefix-matched → decode tells the prefill engine's private coordinator to unpin the
source KV blocks so they are LRU-reusable → decode computes the first output
token and streams tokens back. Each endpoint is launched by one
`python -m megatron.inference.integrations.dynamo` command; individual model-parallel ranks are not
registered with Dynamo.

## Run

Inside the container, after building / staging the model checkpoint:

```bash
# 1. Bring up the stack (blocks; tail logs in another pane).
bash orchestrate.sh

# 2. In another pane:
bash test_phase3.sh
# or
pytest -q test_phase3.py
```

By default both engines run TP=1 → 2 GPUs total. Set `TP_PREFILL` and
`TP_DECODE` to change the complete rank-group sizes.

## Configuration knobs

| Var | Default | Notes |
| --- | --- | --- |
| `MODEL_CHECKPOINT` | `$STAGE/models/llama3.1-8b-instruct-mcore` | mcore-format checkpoint |
| `TOKENIZER_MODEL` | `meta-llama/Llama-3.1-8B-Instruct` | HF id for tokenizer (Instruct variant — base has no `chat_template`, see `[[project_phase0_stack_quirks]]`) |
| `TP_PREFILL` / `TP_DECODE` | `1` | GPUs per TP dimension; may differ when the Megatron revision supports KV re-sharding. |
| `COORD_PORT_PREFILL` / `COORD_PORT_DECODE` | `5555` / `5556` | Coordinator ZMQ. |
| `NIXL_PORT_PREFILL` / `NIXL_PORT_DECODE` | `7000` / `7001` | NIXL listen sockets — must be distinct. |
| `HTTP_PORT` | `8100` | Frontend. |

## What the test asserts

1. `/v1/models` lists the served model.
2. A streaming chat completion returns deltas and finishes. (Tokens coming
   back is strong evidence the whole pipeline works — broken NIXL or broken
   KV import would produce garbage or crash, not a coherent stream.)
3. `DISAGG_PREFILL_HANDOFF request_id=... pinned_blocks=N` is present in
   `worker-prefill.log`. Proves the prefill engine pinned blocks and
   emitted `disaggregated_params`.
4. `DISAGG_DECODE_IMPORT request_id=... prompt_tokens=... imported_blocks=N
   hashes_registered=M` is present in `worker-decode.log`. Proves the
   decode engine took the NIXL-import path **and registered hashes so the
   prefix-cache match could skip prefill**. `hashes_registered > 0` is the
   load-bearing assertion — without it, NIXL might transfer but prefill
   would still re-run.

If only (1) and (2) pass but (3) or (4) fail, the deployment is using the
Phase-0 aggregated path even though the workers report `--role prefill` /
`--role decode`. Most likely cause: the frontend's `PrefillRouter` didn't
activate — check that both workers registered under their expected NATS
subjects (`dynamo.prefill.generate` and `dynamo.backend.generate`).

## Known limitations

- **One node only.** Multi-node will work in principle (NIXL handles RDMA);
  the orchestrator just doesn't lay it out. Extend with srun for that.
- **DP is exactly one per command.** Scale horizontally by adding complete
  Dynamo component replicas.
- **The "partial-block-at-tail" case still does ~1 block of prefill work
  on decode.** Block hashes only cover whole blocks, so any tokens beyond
  the last whole-block boundary go through the normal prefill path. For
  block_size=64 and a 200-token prompt, that's ~8 tokens of redundant
  compute — acceptable.

## What's NOT yet here (follow-ups)

- Phase-1 (per-step metrics PUB) and Phase-2 (KV-event PUB) — stashed under
  `disagg/.stashed_phase1_phase2/`. Independent of Phase-3 and can be
  re-applied any time.
- Block-release before decode starts: the handler emits `RELEASE_KV` only after
  the decode engine returns its first response, which proves the KV pull
  completed. Under LRU prefix caching, released prefill blocks remain cached
  and reusable. A failure during the pull itself can still leave prefill blocks
  pinned until engine restart.
