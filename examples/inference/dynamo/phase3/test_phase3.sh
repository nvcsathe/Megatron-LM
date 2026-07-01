#!/usr/bin/env bash
# Smoke test for an already-up disaggregated stack.

set -euo pipefail

[[ -f /tmp/phase3.env ]] || { echo "FAIL: /tmp/phase3.env not found; is the stack up?" >&2; exit 2; }
source /tmp/phase3.env

echo "[smoke] frontend: $PHASE3_FRONTEND_URL"
echo "[smoke] model:    $PHASE3_MODEL_NAME"

# /v1/models lists the served model.
MODELS=$(curl -sf "$PHASE3_FRONTEND_URL/v1/models")
echo "$MODELS" | grep -q "$PHASE3_MODEL_NAME" \
    || { echo "FAIL: $PHASE3_MODEL_NAME not in /v1/models: $MODELS" >&2; exit 1; }
echo "[smoke] /v1/models OK"

# Drive a prompt long enough to fill multiple KV blocks.
LONG_PROMPT=$(printf 'Hello %.0s' {1..200})
RESPONSE=$(curl -sN "$PHASE3_FRONTEND_URL/v1/chat/completions" \
    -H 'content-type: application/json' \
    -d "{\"model\":\"$PHASE3_MODEL_NAME\",
         \"messages\":[{\"role\":\"user\",\"content\":\"$LONG_PROMPT Summarize.\"}],
         \"stream\":true, \"max_tokens\":32}")

DELTA_COUNT=$(echo "$RESPONSE" | grep -c '"delta"' || true)
FINISH_LINE=$(echo "$RESPONSE" | grep -m1 '"finish_reason":"[^"]*"' || true)
if [[ $DELTA_COUNT -lt 1 ]]; then
    echo "FAIL: no streamed deltas. raw response:" >&2
    echo "$RESPONSE" >&2
    exit 1
fi
if [[ -z $FINISH_LINE ]]; then
    echo "FAIL: no finish_reason. raw response:" >&2
    echo "$RESPONSE" >&2
    exit 1
fi
echo "[smoke] streamed $DELTA_COUNT deltas, finish: $FINISH_LINE"

# Prefill pinned blocks for a handoff request.
sleep 2  # give the engines a beat to flush logs
PREFILL_MARKER=$(grep -m1 'DISAGG_PREFILL_HANDOFF' "$PHASE3_PREFILL_LOG" || true)
if [[ -z $PREFILL_MARKER ]]; then
    echo "FAIL: no DISAGG_PREFILL_HANDOFF in $PHASE3_PREFILL_LOG" >&2
    echo "         (prefill engine never pinned KV blocks; handoff path not taken)" >&2
    exit 1
fi
echo "[smoke] prefill engine: $PREFILL_MARKER"

# Decode imported KV and registered hashes for prefill-skip.
DECODE_MARKER=$(grep -m1 'DISAGG_DECODE_IMPORT' "$PHASE3_DECODE_LOG" || true)
if [[ -z $DECODE_MARKER ]]; then
    echo "FAIL: no DISAGG_DECODE_IMPORT in $PHASE3_DECODE_LOG" >&2
    echo "         (decode engine never took the kv-handoff path)" >&2
    exit 1
fi
echo "[smoke] decode engine:  $DECODE_MARKER"

# hashes_registered > 0 means prefix matching can skip prefill.
HASHES_REG=$(echo "$DECODE_MARKER" | grep -oP 'hashes_registered=\K\d+')
if [[ -z $HASHES_REG || $HASHES_REG -lt 1 ]]; then
    echo "FAIL: decode imported blocks but registered 0 hashes; prefill not skipped" >&2
    echo "         $DECODE_MARKER" >&2
    exit 1
fi

echo "[smoke] PASS: end-to-end disagg verified (hashes_registered=$HASHES_REG)"
