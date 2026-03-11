from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable
from typing import ClassVar as _ClassVar, Optional as _Optional

DESCRIPTOR: _descriptor.FileDescriptor

class ValidateRequest(_message.Message):
    __slots__ = ("client_idx", "req_idx", "input_ids", "position_ids", "cache_seq_indices", "parent_indices", "attention_mask", "prefill", "prefix")
    CLIENT_IDX_FIELD_NUMBER: _ClassVar[int]
    REQ_IDX_FIELD_NUMBER: _ClassVar[int]
    INPUT_IDS_FIELD_NUMBER: _ClassVar[int]
    POSITION_IDS_FIELD_NUMBER: _ClassVar[int]
    CACHE_SEQ_INDICES_FIELD_NUMBER: _ClassVar[int]
    PARENT_INDICES_FIELD_NUMBER: _ClassVar[int]
    ATTENTION_MASK_FIELD_NUMBER: _ClassVar[int]
    PREFILL_FIELD_NUMBER: _ClassVar[int]
    PREFIX_FIELD_NUMBER: _ClassVar[int]
    client_idx: int
    req_idx: int
    input_ids: bytes
    position_ids: bytes
    cache_seq_indices: bytes
    parent_indices: bytes
    attention_mask: bytes
    prefill: bool
    prefix: str
    def __init__(self, client_idx: _Optional[int] = ..., req_idx: _Optional[int] = ..., input_ids: _Optional[bytes] = ..., position_ids: _Optional[bytes] = ..., cache_seq_indices: _Optional[bytes] = ..., parent_indices: _Optional[bytes] = ..., attention_mask: _Optional[bytes] = ..., prefill: bool = ..., prefix: _Optional[str] = ...) -> None: ...

class ValidateResponse(_message.Message):
    __slots__ = ("selection", "prefill")
    SELECTION_FIELD_NUMBER: _ClassVar[int]
    PREFILL_FIELD_NUMBER: _ClassVar[int]
    selection: bytes
    prefill: int
    def __init__(self, selection: _Optional[bytes] = ..., prefill: _Optional[int] = ...) -> None: ...

class SyncRequest(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class SyncResponse(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class VerifyBundleRequest(_message.Message):
    __slots__ = ("client_id", "request_id", "bundle_id", "base_token_index", "prompt_hash", "prefix_hash", "token_ids", "timestamp_send_ms", "draft_model_id", "epoch")
    CLIENT_ID_FIELD_NUMBER: _ClassVar[int]
    REQUEST_ID_FIELD_NUMBER: _ClassVar[int]
    BUNDLE_ID_FIELD_NUMBER: _ClassVar[int]
    BASE_TOKEN_INDEX_FIELD_NUMBER: _ClassVar[int]
    PROMPT_HASH_FIELD_NUMBER: _ClassVar[int]
    PREFIX_HASH_FIELD_NUMBER: _ClassVar[int]
    TOKEN_IDS_FIELD_NUMBER: _ClassVar[int]
    TIMESTAMP_SEND_MS_FIELD_NUMBER: _ClassVar[int]
    DRAFT_MODEL_ID_FIELD_NUMBER: _ClassVar[int]
    EPOCH_FIELD_NUMBER: _ClassVar[int]
    client_id: str
    request_id: str
    bundle_id: int
    base_token_index: int
    prompt_hash: str
    prefix_hash: str
    token_ids: _containers.RepeatedScalarFieldContainer[int]
    timestamp_send_ms: int
    draft_model_id: str
    epoch: int
    def __init__(self, client_id: _Optional[str] = ..., request_id: _Optional[str] = ..., bundle_id: _Optional[int] = ..., base_token_index: _Optional[int] = ..., prompt_hash: _Optional[str] = ..., prefix_hash: _Optional[str] = ..., token_ids: _Optional[_Iterable[int]] = ..., timestamp_send_ms: _Optional[int] = ..., draft_model_id: _Optional[str] = ..., epoch: _Optional[int] = ...) -> None: ...

class VerifyBundleResponse(_message.Message):
    __slots__ = ("request_id", "bundle_id", "base_token_index", "accept_bitmap", "accepted_len", "r_obs", "next_credit", "server_queue_delay_ms", "server_service_ms", "reject_reason", "verifier_poisoned")
    REQUEST_ID_FIELD_NUMBER: _ClassVar[int]
    BUNDLE_ID_FIELD_NUMBER: _ClassVar[int]
    BASE_TOKEN_INDEX_FIELD_NUMBER: _ClassVar[int]
    ACCEPT_BITMAP_FIELD_NUMBER: _ClassVar[int]
    ACCEPTED_LEN_FIELD_NUMBER: _ClassVar[int]
    R_OBS_FIELD_NUMBER: _ClassVar[int]
    NEXT_CREDIT_FIELD_NUMBER: _ClassVar[int]
    SERVER_QUEUE_DELAY_MS_FIELD_NUMBER: _ClassVar[int]
    SERVER_SERVICE_MS_FIELD_NUMBER: _ClassVar[int]
    REJECT_REASON_FIELD_NUMBER: _ClassVar[int]
    VERIFIER_POISONED_FIELD_NUMBER: _ClassVar[int]
    request_id: str
    bundle_id: int
    base_token_index: int
    accept_bitmap: _containers.RepeatedScalarFieldContainer[bool]
    accepted_len: int
    r_obs: float
    next_credit: int
    server_queue_delay_ms: float
    server_service_ms: float
    reject_reason: str
    verifier_poisoned: bool
    def __init__(self, request_id: _Optional[str] = ..., bundle_id: _Optional[int] = ..., base_token_index: _Optional[int] = ..., accept_bitmap: _Optional[_Iterable[bool]] = ..., accepted_len: _Optional[int] = ..., r_obs: _Optional[float] = ..., next_credit: _Optional[int] = ..., server_queue_delay_ms: _Optional[float] = ..., server_service_ms: _Optional[float] = ..., reject_reason: _Optional[str] = ..., verifier_poisoned: bool = ...) -> None: ...
