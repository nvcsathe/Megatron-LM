# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Dynamo-specific extension of Megatron's inference client."""

from megatron.core.inference.headers import Headers
from megatron.core.inference.inference_client import InferenceClient
from megatron.inference.integrations.dynamo.protocol import DynamoClientProtocolMixin
from megatron.inference.integrations.dynamo.telemetry import DynamoClientTelemetryMixin


class DynamoInferenceClient(DynamoClientProtocolMixin, DynamoClientTelemetryMixin, InferenceClient):
    """Megatron inference client with Dynamo management and KV-event support."""

    def __init__(self, inference_coordinator_address: str, deserialize: bool = False) -> None:
        super().__init__(inference_coordinator_address, deserialize=deserialize)
        self._initialize_dynamo_protocol()
        self._initialize_dynamo_telemetry()

    def _handle_extension_message(self, header: Headers, data: list) -> bool:
        return self._handle_dynamo_protocol_message(
            header, data
        ) or self._handle_dynamo_telemetry_message(header, data)

    def _cleanup_extension_state(self) -> None:
        self._cancel_dynamo_protocol_futures()
