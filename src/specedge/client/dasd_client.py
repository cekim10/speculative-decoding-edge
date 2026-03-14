from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class DasdFailureCacheEntry:
    token_id: int
    blocked_until_round: int
    hits: int = 1


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
        raw_delta = accepted_len - rejected_len
        streak_before = {
            "strong_accept_streak": self.strong_accept_streak,
            "full_rejection_streak": self.full_rejection_streak,
        }

        if proposed_len > 0 and accepted_len == proposed_len:
            self.strong_accept_streak += 1
            self.full_rejection_streak = 0
        elif accepted_len == 0 and proposed_len > 0:
            self.full_rejection_streak += 1
            self.strong_accept_streak = 0
        else:
            self.strong_accept_streak = 0
            self.full_rejection_streak = 0

        unclamped_credit = self.credit

        if self.adaptive_enabled:
            unclamped_credit += raw_delta

            if proposed_len > 0 and accepted_len == proposed_len:
                unclamped_credit += self.success_bonus
                if self.strong_accept_streak >= 2 and self.success_bonus > 0:
                    unclamped_credit += 1
            elif accepted_len == 0 and proposed_len > 0:
                unclamped_credit -= self.rejection_penalty

            self.credit = max(self.credit_min, min(self.credit_max, unclamped_credit))
        else:
            unclamped_credit = self.credit

        return {
            "adaptive_enabled": self.adaptive_enabled,
            "credit_before": credit_before,
            "credit_after": self.credit,
            "raw_delta": raw_delta,
            "unclamped_credit": unclamped_credit,
            "clamped_credit": self.credit,
            "accepted_len": accepted_len,
            "proposed_len": proposed_len,
            "rejected_len": rejected_len,
            "strong_accept_streak_before": streak_before["strong_accept_streak"],
            "full_rejection_streak_before": streak_before["full_rejection_streak"],
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
    request_id: str
    bundle_id: int
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
    verify_rounds: int = 0
    full_rejection_count: int = 0
    max_full_rejection_streak: int = 0
    stall_rounds: int = 0
    same_base_retry_count: int = 0
    recovery_mode_entries: int = 0
    failure_cache_hits: int = 0
    forced_commit_count: int = 0
    invalid_token_abort_count: int = 0
    stale_epoch_drop_count: int = 0
    late_task_drop_count: int = 0
    inflight_cleanup_count: int = 0
    rpc_failure_count: int = 0
    rpc_unavailable_abort_count: int = 0
    unexpected_stall_count: int = 0
    refill_attempt_count: int = 0
    refill_success_count: int = 0
    refill_skip_count: int = 0
    rollback_rebuild_count: int = 0
    unexpected_empty_inflight_recovery_count: int = 0
    unexpected_empty_inflight_recovery_fail_count: int = 0
    premature_stall_prevented_count: int = 0
    refill_no_work_count: int = 0
    refill_guard_block_count: int = 0
    bundle_build_none_count: int = 0
    send_spawn_count: int = 0
    send_spawn_after_rebuild_count: int = 0
    draft_regeneration_attempt_count: int = 0
    draft_regeneration_success_count: int = 0
    draft_regeneration_no_suffix_count: int = 0
    recovery_resume_success_count: int = 0
    recovery_resume_fail_count: int = 0
    rollback_due_to_full_rejection_threshold_count: int = 0
    rollback_due_to_same_base_retry_threshold_count: int = 0
    rollback_due_to_failure_cache_blocked_suffix_count: int = 0
    rollback_due_to_forced_commit_failure_count: int = 0
    rollback_due_to_recovery_mode_entry_count: int = 0
    rollback_due_to_post_recovery_rejection_count: int = 0
    rollback_due_to_contiguous_commit_mismatch_count: int = 0
    rollback_due_to_state_inconsistency_count: int = 0
    rollback_due_to_retry_loop_breaker_count: int = 0
    retry_loop_suspected_count: int = 0
    duplicate_retry_fingerprint_count: int = 0
    unique_retry_fingerprint_count: int = 0
    per_base_max_retry_count: int = 0
    per_base_max_full_rejection_count: int = 0
    per_base_max_same_base_retry_count: int = 0
    recovery_mode_entry_count: int = 0
    recovery_mode_exit_count: int = 0
    recovery_mode_success_count: int = 0
    recovery_mode_failure_count: int = 0
    forced_commit_attempt_count: int = 0
    forced_commit_success_count: int = 0
    forced_commit_failure_count: int = 0
    failure_cache_block_decision_count: int = 0
    failure_cache_blocked_same_base_retry_count: int = 0
    local_stabilization_active: bool = False
    local_stabilization_attempt_count: int = 0
    local_stabilization_success_count: int = 0
    local_stabilization_fail_count: int = 0
    suffix_refresh_attempt_count: int = 0
    suffix_refresh_success_count: int = 0
    suffix_refresh_fail_count: int = 0
    rollback_cleanup_deferred_count: int = 0
    rollback_cleanup_escalated_count: int = 0
    rollback_cleanup_avoided_count: int = 0
    cooldown_active: bool = False
    cooldown_entry_count: int = 0
    cooldown_exit_count: int = 0
    cooldown_progress_count: int = 0
    cooldown_progress_remaining: int = 0
    unstable_phase_inflight_cap_hits: int = 0
    low_value_retry_suppressed_count: int = 0
    same_base_retry_short_circuit_count: int = 0
    cheap_recovery_success_count: int = 0
    expensive_recovery_count: int = 0
    sum_window_at_send: int = 0
    sum_tree_depth_at_send: int = 0
    sum_leaf_budget_at_send: int = 0
    finish_status: str = ""
    last_refill_skip_reason: str = ""
    last_recovery_failure_reason: str = ""
    last_refill_phase: str = ""
    last_rollback_cause: str = ""
    last_retry_decision_reason: str = ""
    last_recovery_mode_transition_reason: str = ""
    last_forced_commit_decision_reason: str = ""
    last_mitigation_decision_reason: str = ""
    last_suffix_refresh_reason: str = ""
    last_cooldown_reason: str = ""
    last_retry_quality_reason: str = ""
    suffix_refresh_anchor: tuple[int, int] | None = None
    cleanup_reason: str = ""
    cleanup_induced_drain: bool = False
    base_retry_counts: dict[int, int] = field(default_factory=dict)
    failure_cache: dict[tuple[int, tuple[int, ...]], dict[int, DasdFailureCacheEntry]] = (
        field(default_factory=dict)
    )
    forced_commits_by_base: dict[int, int] = field(default_factory=dict)
    recovery_mode_active: bool = False
    recovery_mode_rounds_left: int = 0
    recovery_mode_reason: str = ""
    fallback_burst_active: bool = False
    fallback_burst_steps_left: int = 0
    fallback_burst_total_steps: int = 0
    fallback_burst_sync_base: Optional[int] = None
    fallback_burst_sync_token_id: Optional[int] = None
    base_lifecycle: dict[int, dict[str, Any]] = field(default_factory=dict)
    retry_fingerprints_by_base: dict[int, dict[Any, int]] = field(default_factory=dict)
    current_base_tracking: Optional[int] = None

    def speculative_tokens(self):
        return max(0, len(self.drafted_tokens) - self.committed_len)
