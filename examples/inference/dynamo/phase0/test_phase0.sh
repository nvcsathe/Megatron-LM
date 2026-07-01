#!/usr/bin/env bash
# Phase-0 smoke test. Runs against an already-up stack (launch_phase0.sh).
# Reads /tmp/phase0.env from inside the container, sends a streaming chat
# completion, asserts at least one delta arrived and a finish_reason was set.
#
# Usage (from inside the container, any pane that attached to --container-name=dmg):
#   bash test_phase0.sh
#
# Exits 0 on pass, non-zero on fail.

set -euo pipefail

[[ -f /tmp/phase0.env ]] || { echo "FAIL: /tmp/phase0.env not found — is the stack up?" >&2; exit 2; }
source /tmp/phase0.env

echo "[smoke] frontend: $PHASE0_FRONTEND_URL"
echo "[smoke] model:    $PHASE0_MODEL_NAME"

# 1. /v1/models lists the served model.
MODELS=$(curl -sf "$PHASE0_FRONTEND_URL/v1/models")
echo "$MODELS" | grep -q "$PHASE0_MODEL_NAME" \
    || { echo "FAIL: $PHASE0_MODEL_NAME not in /v1/models: $MODELS" >&2; exit 1; }
echo "[smoke] /v1/models OK"

# 2. /v1/chat/completions streams ≥1 delta and finishes with stop/length.
RESPONSE=$(curl -sN "$PHASE0_FRONTEND_URL/v1/chat/completions" \
    -H 'content-type: application/json' \
    -d "{\"model\":\"$PHASE0_MODEL_NAME\",
         \"messages\":[{\"role\":\"user\",\"content\":\"Say hi in one short sentence.\"}],
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
echo "[smoke] PASS"
