#!/usr/bin/env bash
# Verify Mamba state transfer in a hybrid disagg stack (Nemotron-3 Nano) brought
# up by orchestrate.sh.
#
# Checks, in order:
#   1. The disagg decode coordinator log shows it IMPORTED KV *and* Mamba state
#      for the request (DISAGG_DECODE_IMPORT + DISAGG_DECODE_MAMBA_IMPORT). If
#      mamba_blocks==0 the prompt was too short to span a committed Mamba block
#      boundary and the transfer path wasn't exercised — treated as FAIL so the
#      test can't silently pass without testing anything.
#   2. GOLD STANDARD: the same greedy (temperature=0) prompt decoded through the
#      disagg stack and through the aggregated (non-disagg) baseline produce the
#      IDENTICAL token text. This is the only check that catches lost/zeroed
#      Mamba state: attention KV transfers correctly regardless, so an
#      attention-only coherence check passes even when conv/ssm state is wrong.
#      Greedy decoding is deterministic, and the KV/Mamba handoff only moves
#      bytes — it does not change the math — so the sequences must match exactly.
#
# If the baseline stack is absent (WITH_BASELINE=0 at orchestrate time), step 2
# degrades to a coherence-only check and prints a warning.
#
# Usage (inside the container, after orchestrate.sh prints PHASE3_MAMBA_READY):
#     source /tmp/phase3_mamba.env
#     ./verify_mamba.sh
#
# Greedy=0 requires the worker to map temperature==0 -> argmax. If your build
# forwards 0 into torch sampling (div-by-zero -> NaN -> CUDA assert), override
# with a fixed seed + positive temperature is NOT sufficient for the exact diff;
# instead patch the worker's argmax fast-path. See verify_hetero_tp.sh notes.

set -uo pipefail
source /tmp/phase3_mamba.env

TEMPERATURE="${TEMPERATURE:-0}"
MAX_TOKENS="${MAX_TOKENS:-64}"

# A prompt long enough to span multiple KV blocks so the prefill side commits at
# least one block-boundary Mamba snapshot to transfer. Override with PROMPT=...
PROMPT="${PROMPT:-$(printf 'You are a careful assistant. %.0s' {1..40})Explain in three sentences why the sky appears blue during the day and red at sunset, and what role Rayleigh scattering plays.}"

echo "== hybrid Mamba disagg verify (temperature=$TEMPERATURE, max_tokens=$MAX_TOKENS) =="

# ---- helper: greedy raw completion -> stdout text --------------------------
# Uses /v1/completions (raw prompt), not /v1/chat/completions: the Nano
# tokenizer is a plain vocab.json with no chat template. Both stacks serve the
# same model, so the prompt is tokenized identically on each.
complete() {
    local url="$1"
    local resp
    resp=$(curl -sf "$url/v1/completions" \
        -H 'Content-Type: application/json' \
        -d "{\"model\":\"$PHASE3_MODEL_NAME\",
             \"prompt\":$(printf '%s' "$PROMPT" | python -c 'import json,sys; print(json.dumps(sys.stdin.read()))'),
             \"max_tokens\":$MAX_TOKENS,\"temperature\":$TEMPERATURE,\"stream\":false}") \
      || { echo "__CURL_FAIL__"; return 1; }
    echo "$resp" | python -c 'import sys,json; print(json.load(sys.stdin)["choices"][0]["text"])'
}

# ---- 1. disagg completion + import markers ---------------------------------
DISAGG_TEXT=$(complete "$PHASE3_FRONTEND_URL") || { echo "FAIL: disagg request errored"; exit 1; }
echo "--- disagg completion ---"
echo "$DISAGG_TEXT"

WORDS=$(echo "$DISAGG_TEXT" | wc -w)
if (( WORDS < 3 )); then
    echo "FAIL: degenerate disagg output ($WORDS words)"; exit 1
fi

if ! grep -q "DISAGG_DECODE_IMPORT" "$PHASE3_DECODE_LOG"; then
    echo "FAIL: no DISAGG_DECODE_IMPORT in $PHASE3_DECODE_LOG — decode re-prefilled instead of importing KV"
    exit 1
fi
MAMBA_LINE=$(grep -m1 "DISAGG_DECODE_MAMBA_IMPORT" "$PHASE3_DECODE_LOG" || true)
if [[ -z "$MAMBA_LINE" ]]; then
    echo "FAIL: no DISAGG_DECODE_MAMBA_IMPORT in $PHASE3_DECODE_LOG — Mamba state was NOT transferred"
    exit 1
fi
echo "$MAMBA_LINE"
MAMBA_BLOCKS=$(echo "$MAMBA_LINE" | grep -oP 'mamba_blocks=\K[0-9]+' || echo 0)
if (( MAMBA_BLOCKS < 1 )); then
    echo "FAIL: mamba_blocks=$MAMBA_BLOCKS — prompt too short to span a committed Mamba block boundary; transfer path untested. Use a longer PROMPT."
    exit 1
fi
echo "PASS: decode imported KV + $MAMBA_BLOCKS Mamba block(s) of conv/ssm state"

# ---- 2. gold-standard token diff vs aggregated baseline --------------------
if [[ -z "${PHASE3_BASELINE_URL:-}" ]]; then
    echo "WARN: no baseline stack (WITH_BASELINE=0). Skipping exact token diff."
    echo "      Re-run orchestrate.sh with WITH_BASELINE=1 for the correctness guarantee."
    echo "RESULT: PARTIAL PASS (import verified, correctness not diffed)"
    exit 0
fi

BASELINE_TEXT=$(complete "$PHASE3_BASELINE_URL") || { echo "FAIL: baseline request errored"; exit 1; }
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
