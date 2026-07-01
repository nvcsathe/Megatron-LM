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
    ENGINE_REPLY_PARTIAL = auto()
    PAUSE = auto()
    UNPAUSE = auto()
    SUSPEND = auto()
    RESUME = auto()
    SET_GENERATION_EPOCH = auto()
    STOP = auto()
    DISCONNECT = auto()
    SHUTDOWN = auto()
    TP_BROADCAST = auto()
    SUBMIT_REQUEST_WITH_KV = auto()  # Decode-side KV import.
    RELEASE_KV = auto()  # Free pinned handoff blocks.
    ABORT_REQUEST = auto()  # Cancel one in-flight request.

    # Dynamo management/telemetry extensions. Their behavior lives under
    # megatron.inference.integrations.dynamo. Keep these entries appended so
    # the numeric values of the established request protocol remain stable.
    GET_METADATA = auto()
    METADATA_REPLY = auto()
    GET_STATUS = auto()
    STATUS_REPLY = auto()
    SUBSCRIBE_TELEMETRY = auto()
    KV_EVENT = auto()
    ENGINE_STATUS = auto()
    CONTROL_ACK = auto()


class UnknownHeaderError(Exception):
    """A signal with an unrecognized header was received by the coordinator."""

    def __init__(self, header):
        super().__init__(f"specialize for {header}.")
