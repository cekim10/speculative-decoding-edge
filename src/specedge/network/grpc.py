from typing import Optional

import grpc.aio
import torch

from specedge_grpc import specedge_pb2, specedge_pb2_grpc
from util import decode, encode


class GrpcClientController:
    def __init__(self, host: str, device: torch.device) -> None:
        self.client_idx = 0

        self._host = host
        self._device = device
        self._channel = grpc.aio.insecure_channel(self._host)
        self._stub = specedge_pb2_grpc.SpecEdgeServiceStub(self._channel)

    async def close(self):
        await self._channel.close()

    async def request(
        self,
        client_idx: int,
        req_idx: int,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        cache_seq_indices: torch.Tensor,
        attention_mask: torch.Tensor,
        parent_indices: torch.Tensor,
        prefill: bool = False,
        prefix: Optional[str] = None,
    ):
        if prefill and prefix is None:
            raise ValueError("Prefix must be provided for prefill requests.")

        input_ids_encoded = encode(input_ids)
        position_ids_encoded = encode(position_ids)
        cache_seq_indices_encoded = encode(cache_seq_indices)
        attention_mask_encoded = encode(attention_mask)
        parent_indices_encoded = encode(parent_indices)

        request = specedge_pb2.ValidateRequest(
            client_idx=client_idx,
            req_idx=req_idx,
            input_ids=input_ids_encoded,
            position_ids=position_ids_encoded,
            cache_seq_indices=cache_seq_indices_encoded,
            parent_indices=parent_indices_encoded,
            attention_mask=attention_mask_encoded,
            prefill=prefill,
            prefix=prefix,
        )

        resp = await self._stub.Validate(request)

        return decode(
            resp.selection,
            device=self._device,
            dtype=torch.long,
            shape=input_ids.size(-1),
        ), resp.prefill

    async def verify_bundle(
        self,
        client_id: str,
        request_id: str,
        bundle_id: int,
        epoch: int,
        base_token_index: int,
        token_ids: list[int],
        timestamp_send_ms: int = 0,
        draft_model_id: str = "",
        recovery_fallback_decode: bool = False,
    ):
        request = specedge_pb2.VerifyBundleRequest(
            client_id=client_id,
            request_id=request_id,
            bundle_id=bundle_id,
            epoch=epoch,
            base_token_index=base_token_index,
            token_ids=token_ids,
            timestamp_send_ms=timestamp_send_ms,
            draft_model_id=draft_model_id,
            recovery_fallback_decode=recovery_fallback_decode,
        )

        return await self._stub.VerifyBundleAsync(request)
