#!/usr/bin/env bash

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export TEST_MODE=baseline
exec bash "$SCRIPT_DIR/launch.sh"

