#!/usr/bin/env bash
# Verify attention KV and Mamba state imports for the Nano v3 test stack.

set -uo pipefail
source /tmp/nano_v3_test.env

TEMPERATURE="${TEMPERATURE:-0}"
MAX_TOKENS="${MAX_TOKENS:-64}"

# Long enough to commit at least one Mamba block snapshot.
PROMPT="${PROMPT:-$(printf 'You are a careful assistant. %.0s' {1..40})Explain in three sentences why the sky appears blue during the day and red at sunset, and what role Rayleigh scattering plays.}"

echo "== hybrid Mamba disagg verify (temperature=$TEMPERATURE, max_tokens=$MAX_TOKENS) =="

complete() {
    local url="$1"
    local resp
    resp=$(curl -sf "$url/v1/completions" \
        -H 'Content-Type: application/json' \
        -d "{\"model\":\"$NANO_V3_MODEL_NAME\",
             \"prompt\":$(printf '%s' "$PROMPT" | python -c 'import json,sys; print(json.dumps(sys.stdin.read()))'),
             \"max_tokens\":$MAX_TOKENS,\"temperature\":$TEMPERATURE,\"stream\":false}") \
      || { echo "__CURL_FAIL__"; return 1; }
    echo "$resp" | python -c 'import sys,json; print(json.load(sys.stdin)["choices"][0]["text"])'
}

DISAGG_TEXT=$(complete "$NANO_V3_FRONTEND_URL") || { echo "FAIL: disagg request errored"; exit 1; }
echo "--- disagg completion ---"
echo "$DISAGG_TEXT"

WORDS=$(echo "$DISAGG_TEXT" | wc -w)
if (( WORDS < 3 )); then
    echo "FAIL: degenerate disagg output ($WORDS words)"; exit 1
fi

if ! grep -q "DISAGG_DECODE_IMPORT" "$NANO_V3_DECODE_LOG"; then
    echo "FAIL: no DISAGG_DECODE_IMPORT in $NANO_V3_DECODE_LOG — decode re-prefilled instead of importing KV"
    exit 1
fi
MAMBA_LINE=$(grep -m1 "DISAGG_DECODE_MAMBA_IMPORT" "$NANO_V3_DECODE_LOG" || true)
if [[ -z "$MAMBA_LINE" ]]; then
    echo "FAIL: no DISAGG_DECODE_MAMBA_IMPORT in $NANO_V3_DECODE_LOG — Mamba state was NOT transferred"
    exit 1
fi
echo "$MAMBA_LINE"
MAMBA_BLOCKS=$(echo "$MAMBA_LINE" | grep -oP 'mamba_blocks=\K[0-9]+' || echo 0)
if (( MAMBA_BLOCKS < 1 )); then
    echo "FAIL: mamba_blocks=$MAMBA_BLOCKS — prompt too short to span a committed Mamba block boundary; transfer path untested. Use a longer PROMPT."
    exit 1
fi
echo "PASS: decode imported KV + $MAMBA_BLOCKS Mamba block(s) of conv/ssm state"

if [[ -z "${NANO_V3_BASELINE_URL:-}" ]]; then
    echo "WARN: no baseline stack (WITH_BASELINE=0). Skipping exact token diff."
    echo "      Re-run orchestrate.sh with WITH_BASELINE=1 for the correctness guarantee."
    echo "RESULT: PARTIAL PASS (import verified, correctness not diffed)"
    exit 0
fi

BASELINE_TEXT=$(complete "$NANO_V3_BASELINE_URL") || { echo "FAIL: baseline request errored"; exit 1; }
echo "--- baseline (aggregated) completion ---"
echo "$BASELINE_TEXT"

if [[ "$DISAGG_TEXT" == "$BASELINE_TEXT" ]]; then
    echo "RESULT: PASS — disagg output is byte-identical to the aggregated baseline."
    echo "        Mamba conv/ssm state transferred correctly across the handoff."
    exit 0
else
    echo "RESULT: FAIL — disagg output DIVERGES from the aggregated baseline."
    echo "        Greedy decoding is deterministic, so divergence means the"
    echo "        transferred Mamba (or KV) state is wrong. Diff:"
    diff <(printf '%s\n' "$BASELINE_TEXT") <(printf '%s\n' "$DISAGG_TEXT") || true
    exit 1
fi
