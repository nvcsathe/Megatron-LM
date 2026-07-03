# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
# syntax=docker/dockerfile:1.7

# Megatron-owned Dynamo backend image.
#
# Build from the Megatron-LM repository root:
#
#   docker build -f megatron/inference/integrations/dynamo/container/Dockerfile \
#     --build-arg DYNAMO_REF=9db71762e6d1a47bff42d55938e0ec27583fcef7 \
#     -t megatron-dynamo:dev .
#
# DYNAMO_REF follows the same convention as the Echo backend:
#   empty                  install ai-dynamo from PyPI transitively
#   v?N.N...               install that ai-dynamo release from PyPI
#   branch, tag, or SHA     build ai-dynamo and ai-dynamo-runtime from git

ARG BASE_IMAGE=nvcr.io/nvidia/pytorch
ARG BASE_IMAGE_TAG=26.04-py3

FROM ${BASE_IMAGE}:${BASE_IMAGE_TAG} AS runtime

# Pinned to the current tip of ai-dynamo/dynamo main. Update this together
# with megatron/inference/integrations/dynamo/dynamo-ref.txt after validating a newer revision.
ARG DYNAMO_REF=9db71762e6d1a47bff42d55938e0ec27583fcef7
ARG DYNAMO_REPO=https://github.com/ai-dynamo/dynamo.git
ARG NIXL_VERSION=0.10.1
ARG NATS_VERSION=v2.10.28
ARG ETCD_VERSION=v3.5.30

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HOME=/home/dynamo \
    MEGATRON_HOME=/opt/megatron-lm \
    VIRTUAL_ENV=/opt/dynamo/venv \
    PATH=/opt/dynamo/venv/bin:/usr/local/bin/etcd:/root/.cargo/bin:${PATH} \
    PYTHONPATH=/opt/megatron-lm:${PYTHONPATH:-} \
    NIXL_PREFIX=/opt/nvidia/nvda_nixl \
    LD_LIBRARY_PATH=/opt/nvidia/nvda_nixl/lib64:/opt/nvidia/nvda_nixl/lib64/plugins:/usr/local/ucx/lib:/usr/local/ucx/lib/ucx:/opt/hpcx/ucx/lib:/usr/local/cuda/lib64:/usr/local/cuda/compat:${LD_LIBRARY_PATH:-}

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        clang \
        cmake \
        curl \
        git \
        libclang-dev \
        libgomp1 \
        librdmacm1 \
        openssh-server \
        pkg-config \
        protobuf-compiler \
        python3-venv \
        rdma-core \
    && rm -rf /var/lib/apt/lists/*

# Keep the all-in-one local examples self-contained. Production deployments
# normally run NATS and etcd as separate services.
RUN set -eux; \
    arch="$(uname -m)"; \
    case "${arch}" in \
        x86_64) nats_arch=amd64; etcd_arch=amd64 ;; \
        aarch64) nats_arch=arm64; etcd_arch=arm64 ;; \
        *) echo "unsupported architecture: ${arch}" >&2; exit 1 ;; \
    esac; \
    curl -fsSL "https://github.com/nats-io/nats-server/releases/download/${NATS_VERSION}/nats-server-${NATS_VERSION}-linux-${nats_arch}.tar.gz" \
        | tar -xz -C /tmp; \
    install -m 0755 "/tmp/nats-server-${NATS_VERSION}-linux-${nats_arch}/nats-server" /usr/local/bin/nats-server; \
    curl -fsSL "https://github.com/etcd-io/etcd/releases/download/${ETCD_VERSION}/etcd-${ETCD_VERSION}-linux-${etcd_arch}.tar.gz" \
        | tar -xz -C /tmp; \
    mkdir -p /usr/local/bin/etcd; \
    install -m 0755 "/tmp/etcd-${ETCD_VERSION}-linux-${etcd_arch}/etcd" /usr/local/bin/etcd/etcd; \
    install -m 0755 "/tmp/etcd-${ETCD_VERSION}-linux-${etcd_arch}/etcdctl" /usr/local/bin/etcd/etcdctl; \
    rm -rf /tmp/nats-server-* /tmp/etcd-*

RUN python3 -m venv --system-site-packages /opt/dynamo/venv \
    && python -m pip install --upgrade pip uv

WORKDIR /opt/megatron-lm
COPY megatron/inference/integrations/dynamo/container/requirements.txt /tmp/megatron-dynamo-requirements.txt
COPY megatron/inference/integrations/dynamo/scripts/install-dynamo.sh /tmp/install-dynamo.sh

# Install Dynamo first so its Python and native runtime are pinned together.
# Install NIXL separately because it is the Megatron disaggregation transport,
# not part of the backend adapter's Python dependency contract. Select the
# CUDA 13 wheel explicitly and do not resolve its generic torch dependency:
# the NGC base already provides a matched Torch/CUDA/Transformer Engine stack.
RUN --mount=type=cache,target=/root/.cache/uv,sharing=locked \
    DYNAMO_REPO="${DYNAMO_REPO}" bash /tmp/install-dynamo.sh "${DYNAMO_REF}" \
    && uv pip install --no-deps \
        "nixl==${NIXL_VERSION}" \
        "nixl-cu13==${NIXL_VERSION}" \
    && uv pip install --no-deps --requirement /tmp/megatron-dynamo-requirements.txt

COPY . /opt/megatron-lm

# The runtime executes directly from this checkout. --no-deps preserves the
# PyTorch/CUDA stack supplied by the NGC base image.
RUN --mount=type=cache,target=/root/.cache/uv,sharing=locked \
    uv pip install --no-deps -e /opt/megatron-lm \
    && uv pip install --no-deps ninja \
    && MAMBA_FORCE_BUILD=TRUE CAUSAL_CONV1D_FORCE_BUILD=TRUE \
       uv pip install --no-deps --no-build-isolation \
           "causal-conv1d~=1.5" "mamba-ssm~=2.2"

RUN userdel -r ubuntu >/dev/null 2>&1 || true; \
    useradd -m -s /bin/bash -u 1000 -g 0 dynamo; \
    mkdir -p /home/dynamo/.cache; \
    chown -R dynamo:0 /home/dynamo /opt/dynamo /opt/megatron-lm

USER dynamo
WORKDIR /opt/megatron-lm

ENTRYPOINT []
CMD ["/bin/bash"]
