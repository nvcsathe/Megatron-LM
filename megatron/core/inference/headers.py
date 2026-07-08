# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

from enum import Enum, auto


class Headers(Enum):
    """
    Enum representing headers used for communication with the inference-coordinator.
    """

    CONNECT = auto()
    CONNECT_ACK = auto()
    SUBMIT_REQUEST = auto()
    ENGINE_REPLY = auto()
    ENGINE_REPLY_PARTIAL = auto()  # Currently used only by the Dynamo frontend.
    PAUSE = auto()
    UNPAUSE = auto()
    SUSPEND = auto()
    RESUME = auto()
    SET_GENERATION_EPOCH = auto()
    STOP = auto()
    DISCONNECT = auto()
    SHUTDOWN = auto()
    TP_BROADCAST = auto()
    START_CUDA_PROFILER = auto()
    STOP_CUDA_PROFILER = auto()
    SUBMIT_REQUEST_WITH_KV = auto()  # Decode-side KV import.
    RELEASE_KV = auto()  # Free pinned handoff blocks.
    ABORT_REQUEST = auto()  # Cancel one in-flight request.
    REGISTER_ROLE = auto()  # Engine announces its disagg role (prefill/decode).
    KV_READ_DONE = auto()  # Decode finished importing a hand-off's KV.
    SEND_KV = auto()  # Push transport: tell the prefill to send a hand-off's KV.


class UnknownHeaderError(Exception):
    """A signal with an unrecognized header was received by the coordinator."""

    def __init__(self, header):
        super().__init__(f"specialize for {header}.")
