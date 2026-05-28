# Dynamo Streaming Tests

This README covers the unit and integration tests that exercise the
`InferenceClient.add_request_streaming` path and the engine-side
`ENGINE_REPLY_PARTIAL` emission added for the Dynamo backend.

The tests below were authored alongside the code change and have **not** been
executed yet — the developer environment does not have the Megatron CI
container available for `torch.distributed.run`. Run them under the CI
container before merging.

## What's covered

- `test_inference_client_streaming.py` — unit tests against a mocked ZMQ
  socket. Verifies the SUBMIT_REQUEST payload carries `streaming=True`, that
  `ENGINE_REPLY_PARTIAL` frames surface as `{"partial": ...}` items in order,
  that the terminal `ENGINE_REPLY` surfaces as `{"final": ...}` and ends the
  iterator, that stray partials for unknown request IDs are dropped, and that
  `stop()` releases open iterators.

## What's not yet covered (follow-up tests to add)

- Engine-side: an end-to-end test that runs `DynamicInferenceEngine` against a
  fake coordinator and asserts the engine sends `ENGINE_REPLY_PARTIAL` frames
  while a streaming request is in flight, with monotonically growing
  `new_tokens` slices and no double-emission of the same token. The existing
  fixtures in `tests/unit_tests/inference/coordinator_test_utils.py` are the
  right starting point.
- Coordinator-side: extend
  `tests/unit_tests/inference/test_data_parallel_inference_coordinator.py` to
  drive `ENGINE_REPLY_PARTIAL` frames through the coordinator and assert they
  are forwarded to the originating client identity without consuming the
  routing entries in `request_id_to_client_id`.

## How to run locally

The unit tests must run under `torch.distributed.run` per the project's
testing skill, even though these particular tests don't touch CUDA. Inside the
CI container:

```bash
# All streaming tests in this file:
uv run python -m torch.distributed.run --nproc-per-node 8 -m pytest -q \
    tests/unit_tests/inference/test_inference_client_streaming.py

# Single test:
uv run python -m torch.distributed.run --nproc-per-node 8 -m pytest -q \
    tests/unit_tests/inference/test_inference_client_streaming.py::test_add_request_streaming_emits_partials_then_final
```

Filter by name during development:

```bash
uv run python -m torch.distributed.run --nproc-per-node 8 -m pytest -q \
    tests/unit_tests/inference -k streaming
```

## How to run in CI

These tests live under `tests/unit_tests/inference/` and are picked up by the
existing unit-test buckets. No new recipe YAML entry is needed.

If we later add a functional test that brings up a coordinator + engine and
drives a real streaming request, that test should land under
`tests/functional_tests/test_cases/inference/dynamo_streaming/` with its own
recipe YAML in `tests/test_utils/recipes/h100/`.

## Manual end-to-end smoke (no Dynamo required)

To sanity-check the streaming path without the Dynamo worker, write a small
script that:

1. Launches `tools/run_dynamic_text_generation_server.py --frontend dynamo
   --tensor-model-parallel-size 1 ...` (Megatron model + tokenizer needed).
2. Parses `MEGATRON_COORDINATOR_ADDR=<addr>` from stdout.
3. Connects an `InferenceClient(addr)`, calls `client.start()`, then
   `async for chunk in client.add_request_streaming(prompt, SamplingParams(...))`
   and prints each `chunk["partial"]["new_tokens"]` as it arrives.

This is the harness Phase 0 of the Dynamo integration depends on.
