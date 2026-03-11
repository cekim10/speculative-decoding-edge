from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class DasdTreeBudget:
    max_beam_len: int
    max_budget: int
    max_n_beams: int
    max_branch_width: int


@dataclass
class DasdCreditController:
    # Research-friendly linear controller:
    # credit increases with accepted tokens and decreases with rejected tokens,
    # then maps to the next DASD window and draft-tree budget.
    adaptive_enabled: bool
    adaptive_window_enabled: bool
    adaptive_tree_budget_enabled: bool
    credit_min: int
    credit_max: int
    credit: int
    rejection_penalty: int
    success_bonus: int
    min_window: int
    max_window: int
    min_tree_depth: int
    max_tree_depth: int
    min_leaf_budget: int
    max_leaf_budget: int
    strong_accept_streak: int = 0
    full_rejection_streak: int = 0

    def apply_feedback(self, accepted_len: int, proposed_len: int):
        credit_before = self.credit
        rejected_len = max(0, proposed_len - accepted_len)

        if not self.adaptive_enabled:
            return {
                "credit_before": credit_before,
                "credit_after": self.credit,
                "accepted_len": accepted_len,
                "proposed_len": proposed_len,
                "rejected_len": rejected_len,
                "strong_accept_streak": self.strong_accept_streak,
                "full_rejection_streak": self.full_rejection_streak,
            }

        self.credit += accepted_len - rejected_len

        if proposed_len > 0 and accepted_len == proposed_len:
            self.strong_accept_streak += 1
            self.full_rejection_streak = 0
            self.credit += self.success_bonus
            if self.strong_accept_streak >= 2 and self.success_bonus > 0:
                self.credit += 1
        elif accepted_len == 0 and proposed_len > 0:
            self.full_rejection_streak += 1
            self.strong_accept_streak = 0
            self.credit -= self.rejection_penalty
        else:
            self.strong_accept_streak = 0
            self.full_rejection_streak = 0

        self.credit = max(self.credit_min, min(self.credit_max, self.credit))

        return {
            "credit_before": credit_before,
            "credit_after": self.credit,
            "accepted_len": accepted_len,
            "proposed_len": proposed_len,
            "rejected_len": rejected_len,
            "strong_accept_streak": self.strong_accept_streak,
            "full_rejection_streak": self.full_rejection_streak,
        }

    def current_window(self, fallback: int):
        if not (self.adaptive_enabled and self.adaptive_window_enabled):
            return fallback
        return self._map_range(
            self.credit,
            self.credit_min,
            self.credit_max,
            self.min_window,
            self.max_window,
        )

    def current_tree_budget(
        self,
        fallback_depth: int,
        fallback_leaf_budget: int,
        fallback_max_n_beams: int,
        fallback_max_branch_width: int,
    ):
        if not (self.adaptive_enabled and self.adaptive_tree_budget_enabled):
            return DasdTreeBudget(
                max_beam_len=fallback_depth,
                max_budget=fallback_leaf_budget,
                max_n_beams=fallback_max_n_beams,
                max_branch_width=fallback_max_branch_width,
            )

        max_beam_len = self._map_range(
            self.credit,
            self.credit_min,
            self.credit_max,
            self.min_tree_depth,
            self.max_tree_depth,
        )
        max_beam_len = max(1, min(fallback_depth, max_beam_len))
        max_budget = self._map_range(
            self.credit,
            self.credit_min,
            self.credit_max,
            self.min_leaf_budget,
            self.max_leaf_budget,
        )
        max_budget = max(1, min(fallback_leaf_budget, max_budget))

        return DasdTreeBudget(
            max_beam_len=max_beam_len,
            max_budget=max_budget,
            max_n_beams=max(1, min(fallback_max_n_beams, max_budget)),
            max_branch_width=max(1, min(fallback_max_branch_width, max_budget)),
        )

    def _map_range(
        self,
        value: int,
        src_min: int,
        src_max: int,
        dst_min: int,
        dst_max: int,
    ):
        if src_max <= src_min:
            return dst_min
        clamped = max(src_min, min(src_max, value))
        ratio = (clamped - src_min) / (src_max - src_min)
        mapped = dst_min + ratio * (dst_max - dst_min)
        return int(round(mapped))


@dataclass
class DasdTaskInfo:
    task: Any
    base_token_index: int
    token_ids: list[int]
    epoch_at_send: int
    inflight_at_send: int
    send_ts: float
    window_size_at_send: int
    credit_at_send: int
    tree_depth_at_send: int
    leaf_budget_at_send: int
    branch_width_at_send: int


@dataclass
class DasdBundleResult:
    response: Any
    recv_ts: float
    task_info: DasdTaskInfo
    control_feedback: dict[str, Any] | None = None


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
    rollback_blocked_committed_len: Optional[int] = None
    rollback_blocked_token_id: Optional[int] = None
    credit_controller: DasdCreditController | None = None
    tree_budget: DasdTreeBudget | None = None

    def speculative_tokens(self):
        return max(0, len(self.drafted_tokens) - self.committed_len)
