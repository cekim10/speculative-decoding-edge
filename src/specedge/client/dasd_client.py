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
    # Async DASD controller:
    # credit tracks how deep the speculative pipeline should be given
    # acceptance quality, verifier RTT, and current inflight overlap.
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
    target_acceptance_ratio: float = 0.7
    alpha: float = 2.0
    rtt_target_ms: float = 40.0
    rtt_gain: float = 0.75
    inflight_gain: float = 0.5
    acceptance_weight: float = 1.0
    instability_weight: float = 0.5
    tau_high: float = 0.55
    tau_low: float = 0.1
    tau_critical: float = -0.35
    acceptance_ratio_ema: float = 1.0
    rtt_ema_ms: float = 0.0
    inflight_ema: float = 0.0
    strong_accept_streak: int = 0
    full_rejection_streak: int = 0

    def apply_feedback(
        self,
        accepted_len: int,
        proposed_len: int,
        *,
        rtt_ms: float = 0.0,
        inflight_count: int = 0,
        rollback_distance: int = 0,
        recovery_active: bool = False,
    ):
        credit_before = self.credit
        rejected_len = max(0, proposed_len - accepted_len)
        raw_delta = accepted_len - rejected_len
        acceptance_ratio = (
            float(accepted_len) / float(proposed_len) if proposed_len > 0 else 0.0
        )
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

        unclamped_credit = float(self.credit)
        controller_delta = 0
        rtt_pressure = 0.0
        instability_pressure = 0.0
        health_score = 0.0

        if self.adaptive_enabled:
            self.acceptance_ratio_ema = 0.8 * self.acceptance_ratio_ema + 0.2 * acceptance_ratio
            self.rtt_ema_ms = 0.8 * self.rtt_ema_ms + 0.2 * max(0.0, float(rtt_ms))
            self.inflight_ema = 0.8 * self.inflight_ema + 0.2 * max(0, inflight_count)
            rtt_pressure = self._compute_rtt_pressure(float(rtt_ms))
            instability_pressure = self._compute_instability_pressure(
                acceptance_ratio=acceptance_ratio,
                accepted_len=accepted_len,
                proposed_len=proposed_len,
                rollback_distance=rollback_distance,
                recovery_active=recovery_active,
            )
            health_score = (
                self.acceptance_weight * acceptance_ratio
                - self.rtt_gain * rtt_pressure
                - self.instability_weight * instability_pressure
            )

            if health_score > self.tau_high:
                controller_delta = 1
            elif health_score >= self.tau_low:
                controller_delta = 0
            elif health_score >= self.tau_critical:
                controller_delta = -1
            else:
                controller_delta = -2

            unclamped_credit = float(self.credit) + float(controller_delta)
            if accepted_len == 0:
                unclamped_credit = min(float(self.credit), unclamped_credit)

            self.credit = max(
                self.credit_min,
                min(self.credit_max, int(round(unclamped_credit))),
            )
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
            "acceptance_ratio": acceptance_ratio,
            "acceptance_ratio_ema": self.acceptance_ratio_ema,
            "rtt_ms": float(rtt_ms),
            "rtt_ema_ms": self.rtt_ema_ms,
            "inflight_count": inflight_count,
            "inflight_ema": self.inflight_ema,
            "rollback_distance": rollback_distance,
            "recovery_active": recovery_active,
            "rtt_pressure": rtt_pressure,
            "instability_pressure": instability_pressure,
            "health_score": health_score,
            "controller_delta": controller_delta,
            "strong_accept_streak_before": streak_before["strong_accept_streak"],
            "full_rejection_streak_before": streak_before["full_rejection_streak"],
            "strong_accept_streak": self.strong_accept_streak,
            "full_rejection_streak": self.full_rejection_streak,
        }

    def _compute_rtt_pressure(self, rtt_ms: float):
        if self.rtt_target_ms <= 0:
            return 0.0
        return max(
            0.0,
            min(
                2.0,
                (max(0.0, rtt_ms) - self.rtt_target_ms)
                / max(1.0, 2.0 * self.rtt_target_ms),
            ),
        )

    def _compute_instability_pressure(
        self,
        *,
        acceptance_ratio: float,
        accepted_len: int,
        proposed_len: int,
        rollback_distance: int,
        recovery_active: bool,
    ):
        pressure = 0.0
        if proposed_len > 0 and accepted_len == 0:
            pressure += 1.0
        elif proposed_len > 0 and acceptance_ratio < 0.25:
            pressure += 0.5
        elif proposed_len > 0 and acceptance_ratio < 0.5:
            pressure += 0.2
        if rollback_distance > 0:
            pressure += min(1.0, float(rollback_distance) / 16.0)
        if recovery_active:
            pressure += 1.1
        return min(2.5, pressure)

    def current_window(self, fallback: int, effective_credit: Optional[int] = None):
        if not (self.adaptive_enabled and self.adaptive_window_enabled):
            return fallback
        credit = self.credit if effective_credit is None else effective_credit
        if credit <= self.credit_min:
            return 1
        return self._map_range(
            credit,
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
        effective_credit: Optional[int] = None,
    ):
        if not (self.adaptive_enabled and self.adaptive_tree_budget_enabled):
            return DasdTreeBudget(
                max_beam_len=fallback_depth,
                max_budget=fallback_leaf_budget,
                max_n_beams=fallback_max_n_beams,
                max_branch_width=fallback_max_branch_width,
            )

        credit = self.credit if effective_credit is None else effective_credit
        max_beam_len = self._map_range(
            credit,
            self.credit_min,
            self.credit_max,
            self.min_tree_depth,
            self.max_tree_depth,
        )
        max_beam_len = max(1, min(fallback_depth, max_beam_len))
        max_budget = self._map_range(
            credit,
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

    def current_inflight_cap(self, fallback: int, effective_credit: Optional[int] = None):
        if not self.adaptive_enabled:
            return fallback
        credit = self.credit if effective_credit is None else effective_credit
        return max(
            1,
            min(
                fallback,
                self._map_range(
                    credit,
                    self.credit_min,
                    self.credit_max,
                    1,
                    fallback,
                ),
            ),
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
    frontier_sync_active: bool = False
    frontier_sync_target_committed_len: int | None = None
    frontier_sync_progress_remaining: int = 0
    frontier_sync_entry_count: int = 0
    frontier_sync_exit_count: int = 0
    frontier_sync_success_count: int = 0
    frontier_sync_fail_count: int = 0
    frontier_sync_send_gate_block_count: int = 0
    frontier_sync_inflight_cap_hits: int = 0
    frontier_sync_resume_count: int = 0
    pipeline_resume_grace_active: bool = False
    pipeline_resume_grace_sends_remaining: int = 0
    pipeline_resume_grace_reason: str = ""
    pipeline_resume_grace_entry_count: int = 0
    pipeline_resume_grace_exit_count: int = 0
    pipeline_resume_grace_success_count: int = 0
    pipeline_resume_grace_fail_count: int = 0
    recovery_resend_relax_active: bool = False
    recovery_resend_relax_attempts_remaining: int = 0
    recovery_resend_relax_reason: str = ""
    recovery_resend_relax_entry_count: int = 0
    recovery_resend_relax_exit_count: int = 0
    recovery_resend_relax_used_count: int = 0
    recovery_resend_relax_block_bypass_count: int = 0
    healthy_pipeline_active: bool = False
    healthy_pipeline_entry_count: int = 0
    healthy_pipeline_exit_count: int = 0
    healthy_pipeline_send_count: int = 0
    alternative_frontier_search_count: int = 0
    alternative_frontier_search_success_count: int = 0
    alternative_frontier_search_fail_count: int = 0
    alternative_frontier_candidate_reject_count: int = 0
    alternative_frontier_blocked_first_token_count: int = 0
    alternative_frontier_rebuild_count: int = 0
    alternative_frontier_last_reason: str = ""
    alternative_frontier_last_committed_len: int | None = None
    alternative_frontier_last_blocked_token_id: int | None = None
    alternative_frontier_last_candidate_first_tokens: tuple[int, ...] | None = None
    alternative_frontier_distinct_first_token_target: int = 0
    alternative_frontier_distinct_first_token_last_seen: tuple[int, ...] | None = None
    alternative_frontier_distinct_first_token_search_count: int = 0
    alternative_frontier_distinct_first_token_success_count: int = 0
    alternative_frontier_distinct_first_token_fail_count: int = 0
    alternative_frontier_distinct_first_token_reject_count: int = 0
    alternative_frontier_distinct_first_token_blocked_reject_count: int = 0
    alternative_frontier_distinct_first_token_inspected_count: int = 0
    alternative_frontier_last_distinct_reason: str = ""
    frontier_local_tiny_rebuild_count: int = 0
    frontier_local_tiny_rebuild_success_count: int = 0
    frontier_local_tiny_rebuild_fail_count: int = 0
    frontier_local_tiny_rebuild_candidate_reject_count: int = 0
    frontier_local_tiny_rebuild_blocked_reject_count: int = 0
    frontier_local_tiny_rebuild_inspected_count: int = 0
    frontier_local_tiny_rebuild_repeated_signature_fail_count: int = 0
    frontier_local_tiny_rebuild_last_reason: str = ""
    frontier_local_tiny_rebuild_last_committed_len: int | None = None
    frontier_local_tiny_rebuild_last_blocked_token_id: int | None = None
    frontier_local_tiny_rebuild_last_distinct_signature: tuple[int, ...] | None = None
    frontier_local_tiny_rebuild_root_child_reserve_count: int = 0
    frontier_local_tiny_rebuild_root_child_reserve_success_count: int = 0
    frontier_local_tiny_rebuild_root_child_reserve_fail_count: int = 0
    frontier_local_tiny_rebuild_root_child_reserved_pool_size_total: int = 0
    frontier_local_tiny_rebuild_root_child_blocked_reject_count: int = 0
    frontier_local_tiny_rebuild_root_child_duplicate_reject_count: int = 0
    frontier_local_tiny_rebuild_root_child_inspected_count: int = 0
    frontier_local_tiny_rebuild_root_child_last_signature: tuple[int, ...] | None = None
    frontier_local_tiny_rebuild_root_child_last_reason: str = ""
    frontier_local_tiny_rebuild_root_step_mask_count: int = 0
    frontier_local_tiny_rebuild_root_step_mask_success_count: int = 0
    frontier_local_tiny_rebuild_root_step_mask_fail_count: int = 0
    frontier_local_tiny_rebuild_root_step_mask_blocked_token_total: int = 0
    frontier_local_tiny_rebuild_root_step_mask_filtered_candidate_count: int = 0
    frontier_local_tiny_rebuild_root_step_mask_all_blocked_fail_count: int = 0
    frontier_local_tiny_rebuild_root_step_mask_last_reason: str = ""
    hard_stabilization_active: bool = False
    hard_stabilization_remaining_sends: int = 0
    hard_stabilization_reason: str = ""
    hard_stabilization_entry_count: int = 0
    hard_stabilization_exit_count: int = 0
    hard_stabilization_success_count: int = 0
    hard_stabilization_fail_count: int = 0
    last_rollback_distance: int = 0
    max_rollback_distance: int = 0
    rollback_distance_penalty_count: int = 0
    rollback_distance_penalty_total: int = 0
    rollback_distance_last_reason: str = ""
    instability_credit_clamp_active: bool = False
    instability_credit_clamp_remaining: int = 0
    instability_credit_cap: int | None = None
    instability_credit_clamp_entry_count: int = 0
    instability_credit_clamp_exit_count: int = 0
    instability_growth_clamp_count: int = 0
    regeneration_instability_clamp_count: int = 0
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
    suppressed_retry_loop_count: int = 0
    suppressed_retry_loop_break_count: int = 0
    suppressed_retry_loop_escalation_count: int = 0
    suppressed_retry_last_committed_len: int | None = None
    suppressed_retry_last_fingerprint: tuple[int, ...] | None = None
    suppressed_retry_last_blocked_tokens: tuple[int, ...] | None = None
    last_suppressed_retry_reason: str = ""
    suffix_refresh_last_attempt_key: tuple[int, int] | None = None
    tiny_rebuild_last_attempt_key: tuple[int, int] | None = None
    suffix_refresh_guard_skip_count: int = 0
    tiny_rebuild_guard_skip_count: int = 0
    conservative_forward_entry_count: int = 0
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
    last_frontier_sync_reason: str = ""
    last_pipeline_resume_reason: str = ""
    last_recovery_resend_relax_reason: str = ""
    last_conservative_forward_reason: str = ""
    last_hard_stabilization_reason: str = ""
    last_rollback_distance_penalty_reason: str = ""
    last_instability_clamp_reason: str = ""
    suffix_refresh_anchor_committed_len: int | None = None
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
