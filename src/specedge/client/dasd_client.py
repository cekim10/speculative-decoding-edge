from dataclasses import dataclass, field
from typing import Any


@dataclass
class DasdTaskInfo:
    task: Any
    base_token_index: int
    token_ids: list[int]
    epoch_at_send: int
    inflight_at_send: int
    send_ts: float


@dataclass
class DasdBundleResult:
    response: Any
    recv_ts: float
    task_info: DasdTaskInfo


@dataclass
class DasdRequestState:
    req_idx: int
    request_id: str
    epoch: int = 0
    committed_len: int = 0
    next_base_index: int = 0
    next_bundle_id: int = 0
    window_size: int = 1
    drafted_tokens: list[int] = field(default_factory=list)
    inflight: dict[int, DasdTaskInfo] = field(default_factory=dict)
    task_to_bundle: dict[Any, int] = field(default_factory=dict)
    responses_by_base: dict[int, DasdBundleResult] = field(default_factory=dict)
    total_verified_tokens: int = 0
    total_accepted_tokens: int = 0
    rollbacks_count: int = 0
    consecutive_full_rejections: int = 0
    consecutive_rpc_failures: int = 0
    aborted: bool = False
    abort_reason: str = ""

    def speculative_tokens(self):
        return max(0, len(self.drafted_tokens) - self.committed_len)
