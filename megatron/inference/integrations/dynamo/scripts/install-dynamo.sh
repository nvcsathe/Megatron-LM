#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

# Install ai-dynamo and its native runtime into the active Python environment.
# Classification mirrors the Echo backend:
#   empty                  PyPI, resolved transitively
#   v?N.N...               exact PyPI release
#   branch, tag, or SHA     git source build

set -euo pipefail

REF="${1:-}"
REPO="${DYNAMO_REPO:-https://github.com/ai-dynamo/dynamo.git}"

if [[ -z "${REF}" ]]; then
    echo "ai-dynamo: latest PyPI release"
    python -m pip install ai-dynamo
    exit 0
fi

if [[ "${REF}" =~ ^v?[0-9]+\.[0-9]+ ]]; then
    VERSION="${REF#v}"
    echo "ai-dynamo: PyPI ==${VERSION}"
    python -m pip install "ai-dynamo==${VERSION}"
    exit 0
fi

echo "ai-dynamo: git ${REF}"
if ! command -v cargo >/dev/null 2>&1; then
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
        | sh -s -- -y --default-toolchain stable --profile minimal
    export PATH="${HOME}/.cargo/bin:${PATH}"
fi

python -m pip install \
    "ai-dynamo-runtime @ git+${REPO}@${REF}#subdirectory=lib/bindings/python" \
    "ai-dynamo @ git+${REPO}@${REF}"
