import asyncio
import socket
import time
from contextlib import contextmanager
from types import SimpleNamespace
from typing import Optional

import grpc
import numpy as np
import torch

import log
import util
from config import SpecEdgeClientConfig as config
from specedge.client.dasd_client import (
    DasdBundleResult,
    DasdCreditController,
    DasdFailureCacheEntry,
    DasdRequestState,
    DasdTaskInfo,
    DasdTreeBudget,
)
from specedge.client.proactive import SpecExecProactiveDraft
from specedge.network.grpc import GrpcClientController
from specedge.tree import Tree


class SpecExecClient:
    def __init__(
        self,
        engine,
        tokenizer,
        prompt: str,
        max_len: int,
    ) -> None:
        # logging
        self._logger = log.get_logger()
        self._result_logger = log.get_result_logger()

        self._logger.debug("Initializing SpecExecClient")

        self._optimization = config.optimization
        self._draft_forward_time_mode = (
            "no-sync" if self._optimization >= 2 else "event"
        )
        self._target_time_mode = "no-sync" if self._optimization >= 2 else "sync"

        self._device = config.device
        self._dtype = config.dtype

        self._max_n_beams = config.max_n_beams
        self._max_beam_len = config.max_beam_len
        self._max_branch_width = config.max_branch_width
        self._max_budget = config.max_budget
        self._fixed_max_n_beams = self._max_n_beams
        self._fixed_max_beam_len = self._max_beam_len
        self._fixed_max_branch_width = self._max_branch_width
        self._fixed_max_budget = self._max_budget

        self._proactive_type = config.proactive_type

        self._max_new_tokens = config.max_new_tokens
        self._client_idx = config.client_idx
        self._mode = config.mode
        self._dasd_enabled = self._mode == "dasd" and config.dasd_enable_async
        self._dasd_client_id = f"{config.process_name}@{socket.gethostname()}"
        self._dasd_server_poisoned = False

        self._verify_configs()

        self._engine = engine
        self._tokenizer = tokenizer
        self._engine.reset()

        self._prompt = prompt
        self._prefix_tokens = self._tokenizer.encode(prompt, return_tensors="pt").to(
            self._device
        )[: config.max_len]
        self._initial_prefix_tokens = self._prefix_tokens.clone()
        self._num_original_tokens = self._prefix_tokens.numel()
        self._max_len = max_len

        self._tree = Tree(
            prefix_tokens=self._prefix_tokens,
            device=self._device,
            dtype=self._dtype,
            max_len=self._engine.max_len,
        )
        self._validator = GrpcClientController(host=config.host, device=self._device)
        self._effective_vocab_size = self._resolve_effective_vocab_size()

        self._proactive_client: Optional[SpecExecProactiveDraft] = None
        if self._proactive_type != "disabled" and not self._dasd_enabled:
            self._proactive_client = SpecExecProactiveDraft(
                tree=self._tree,
                engine=self._engine,
                max_len=self._max_len,
            )

            # Whether Proactive Draft was executed in the previous iter
            self._previous_proactive_draft = False

            # Whether Proactive Draft is executed in the current iter
            self._proactive_draft = False

    def _resolve_effective_vocab_size(self):
        # Prefer the model output dimension over tokenizer metadata because
        # speculative draft tokens must be valid for the verifier logits space.
        candidates: list[int] = []
        output_embeddings = None
        try:
            output_embeddings = self._engine._model.get_output_embeddings()
        except Exception:
            output_embeddings = None
        if output_embeddings is not None and getattr(output_embeddings, "weight", None) is not None:
            candidates.append(int(output_embeddings.weight.size(0)))
        lm_head = getattr(self._engine._model, "lm_head", None)
        if lm_head is not None:
            out_features = getattr(lm_head, "out_features", None)
            if out_features is not None:
                candidates.append(int(out_features))
            elif getattr(lm_head, "weight", None) is not None:
                candidates.append(int(lm_head.weight.size(0)))
        model_vocab = getattr(getattr(self._engine._model, "config", None), "vocab_size", None)
        if model_vocab is not None:
            candidates.append(int(model_vocab))
        tokenizer_vocab = getattr(self._tokenizer, "vocab_size", None)
        if tokenizer_vocab is not None:
            candidates.append(int(tokenizer_vocab))
        try:
            candidates.append(int(len(self._tokenizer)))
        except Exception:
            pass
        return next((value for value in candidates if value > 0), None)

    def _validate_dasd_token_ids(
        self,
        state: DasdRequestState,
        token_ids: list[int],
        source_path: str,
        bundle_id: Optional[int] = None,
        base_token_index: Optional[int] = None,
    ):
        vocab_size = self._effective_vocab_size
        for raw_token_id in token_ids:
            if not isinstance(raw_token_id, int):
                self._logger.warning(
                    "[DASD] invalid_token req=%s epoch=%d bundle=%s base=%s source=%s token_id=%r vocab_size=%s reason=non_int",
                    state.request_id,
                    state.epoch,
                    bundle_id,
                    base_token_index,
                    source_path,
                    raw_token_id,
                    vocab_size,
                )
                return False
            if raw_token_id < 0 or (vocab_size is not None and raw_token_id >= vocab_size):
                self._logger.warning(
                    "[DASD] invalid_token req=%s epoch=%d bundle=%s base=%s source=%s token_id=%d vocab_size=%s reason=out_of_range",
                    state.request_id,
                    state.epoch,
                    bundle_id,
                    base_token_index,
                    source_path,
                    raw_token_id,
                    vocab_size,
                )
                return False
        return True

    def _schedule_invalid_token_abort(
        self,
        state: DasdRequestState,
        source_path: str,
        bundle_id: Optional[int] = None,
        base_token_index: Optional[int] = None,
    ):
        state.invalid_token_abort_count += 1
        state.aborted = True
        state.abort_reason = "invalid_token_id"
        state.finish_status = "explicit_abort_reason"
        self._logger.warning(
            "[DASD] invalid_token_abort req=%s epoch=%d bundle=%s base=%s source=%s",
            state.request_id,
            state.epoch,
            bundle_id,
            base_token_index,
            source_path,
        )
        cleanup_task = asyncio.create_task(
            self._abort_dasd_request(state, reason=state.abort_reason)
        )
        cleanup_task.add_done_callback(lambda _: None)

    def _dasd_eos_seen(self, state: DasdRequestState):
        return (
            state.committed_len > 0
            and state.committed_len <= len(state.drafted_tokens)
            and state.drafted_tokens[state.committed_len - 1] == self._tokenizer.eos_token_id
        )

    def _log_finish_condition(self, state: DasdRequestState, reason: str):
        if config.dasd_debug:
            self._logger.info(
                "[DASD] finish_condition_evaluated req=%s epoch=%d reason=%s committed_len=%d drafted_len=%d next_base_index=%d inflight_count=%d outstanding_task_count=%d max_new_tokens=%d eos_seen=%s aborted=%s",
                state.request_id,
                state.epoch,
                reason,
                state.committed_len,
                len(state.drafted_tokens),
                state.next_base_index,
                len(state.inflight),
                len(state.task_to_bundle),
                self._max_new_tokens,
                self._dasd_eos_seen(state),
                state.aborted,
            )

    def _log_refill_decision(
        self,
        state: DasdRequestState,
        phase: str,
        action: str,
        reason: str,
        **extra,
    ):
        state.last_refill_phase = phase
        if action in {"skipped", "no_work", "guard_blocked", "fail"}:
            state.last_refill_skip_reason = reason
        if action == "fail":
            state.last_recovery_failure_reason = reason
        if action == "no_work":
            state.refill_no_work_count += 1
        if action in {"skipped", "guard_blocked"}:
            state.refill_guard_block_count += 1

        if not config.dasd_debug:
            return

        self._logger.info(
            "[DASD] refill_decision req=%s epoch=%d phase=%s action=%s reason=%s committed_len=%d drafted_len=%d next_base_index=%d inflight_count=%d pending_response_count=%d active_task_count=%d credit=%s W=%d tree_depth=%s leaf_budget=%s recovery_mode_active=%s full_rejection_streak=%d same_base_retry_count=%d failure_cache_enabled=%s blocked_tokens=%s aborted=%s finish_status=%s max_new_tokens=%d %s",
            state.request_id,
            state.epoch,
            phase,
            action,
            reason,
            state.committed_len,
            len(state.drafted_tokens),
            state.next_base_index,
            len(state.inflight),
            len(state.responses_by_base),
            len(state.task_to_bundle),
            state.credit_controller.credit if state.credit_controller is not None else None,
            state.window_size,
            state.tree_budget.max_beam_len if state.tree_budget is not None else None,
            state.tree_budget.max_budget if state.tree_budget is not None else None,
            state.recovery_mode_active,
            state.consecutive_full_rejections,
            state.same_base_retry_count,
            config.dasd_failure_cache_enabled,
            sorted(self._compute_blocked_tokens_for_prefix(state)),
            state.aborted,
            state.finish_status,
            self._max_new_tokens,
            " ".join(f"{key}={value}" for key, value in extra.items()),
        )

    def _log_dasd_state_event(
        self,
        event: str,
        state: DasdRequestState,
        decision_reason: str,
        **extra,
    ):
        if not config.dasd_debug:
            return
        self._logger.info(
            "[DASD] %s req=%s epoch=%d base_token_index=%d committed_len=%d drafted_len=%d next_base_index=%d inflight_count=%d pending_response_count=%d active_task_count=%d W=%d tree_depth=%s leaf_budget=%s credit=%s full_rejection_streak=%d same_base_retry_count=%d recovery_mode_active=%s forced_commit_eligible=%s forced_commit_applied=%s verifier_next_token_id=%s reject_reason=%s blocked_tokens=%s decision_reason=%s %s",
            event,
            state.request_id,
            state.epoch,
            state.committed_len,
            state.committed_len,
            len(state.drafted_tokens),
            state.next_base_index,
            len(state.inflight),
            len(state.responses_by_base),
            len(state.task_to_bundle),
            state.window_size,
            state.tree_budget.max_beam_len if state.tree_budget is not None else None,
            state.tree_budget.max_budget if state.tree_budget is not None else None,
            state.credit_controller.credit if state.credit_controller is not None else None,
            state.consecutive_full_rejections,
            state.same_base_retry_count,
            state.recovery_mode_active,
            extra.pop("forced_commit_eligible", None),
            extra.pop("forced_commit_applied", None),
            extra.pop("verifier_next_token_id", None),
            extra.pop("reject_reason", None),
            # Blocked-token lookup used by DASD structured logging must stay
            # non-logging, otherwise the logger recursively re-enters itself.
            sorted(self._compute_blocked_tokens_for_prefix(state)),
            decision_reason,
            " ".join(f"{key}={value}" for key, value in extra.items()),
        )

    def _ensure_base_lifecycle(self, state: DasdRequestState, base_token_index: int):
        lifecycle = state.base_lifecycle.get(base_token_index)
        if lifecycle is None:
            lifecycle = {
                "first_epoch_seen": state.epoch,
                "last_epoch_seen": state.epoch,
                "send_attempts": 0,
                "accepted_tokens_accumulated": 0,
                "full_rejections": 0,
                "forced_commit_attempts": 0,
                "forced_commit_successes": 0,
                "recovery_mode_attempts": 0,
                "rollback_events": 0,
                "unique_retry_fingerprints": set(),
                "retry_fingerprints": {},
                "progressed": False,
                "abandoned": False,
                "same_base_retry_events": 0,
            }
            state.base_lifecycle[base_token_index] = lifecycle
        lifecycle["last_epoch_seen"] = state.epoch
        return lifecycle

    def _retry_fingerprint(
        self,
        state: DasdRequestState,
        base_token_index: int,
        token_ids: list[int],
    ):
        return (
            base_token_index,
            state.window_size,
            state.tree_budget.max_beam_len if state.tree_budget is not None else None,
            state.tree_budget.max_budget if state.tree_budget is not None else None,
            tuple(token_ids[: min(len(token_ids), 8)]),
            len(token_ids),
        )

    def _maybe_emit_base_lifecycle_summary(
        self,
        state: DasdRequestState,
        base_token_index: Optional[int],
        final: bool = False,
        reason: str = "",
    ):
        if base_token_index is None:
            return
        lifecycle = state.base_lifecycle.get(base_token_index)
        if lifecycle is None:
            return
        unique_retry_fingerprints = len(lifecycle["unique_retry_fingerprints"])
        self._log_dasd_state_event(
            "base_lifecycle_summary",
            state,
            decision_reason=reason or ("request_end" if final else "base_transition"),
            base_token_index=base_token_index,
            first_epoch_seen=lifecycle["first_epoch_seen"],
            last_epoch_seen=lifecycle["last_epoch_seen"],
            send_attempts=lifecycle["send_attempts"],
            accepted_tokens_accumulated=lifecycle["accepted_tokens_accumulated"],
            full_rejections=lifecycle["full_rejections"],
            forced_commit_attempts=lifecycle["forced_commit_attempts"],
            forced_commit_successes=lifecycle["forced_commit_successes"],
            recovery_mode_attempts=lifecycle["recovery_mode_attempts"],
            rollback_events=lifecycle["rollback_events"],
            unique_retry_fingerprints=unique_retry_fingerprints,
            progressed=lifecycle["progressed"],
            abandoned=lifecycle["abandoned"],
        )

    def _conservative_dasd_tree_budget(self):
        return DasdTreeBudget(
            max_beam_len=max(1, min(self._fixed_max_beam_len, config.dasd_recovery_forced_tree_depth)),
            max_budget=max(1, min(self._fixed_max_budget, config.dasd_recovery_forced_leaf_budget)),
            max_n_beams=max(1, min(self._fixed_max_n_beams, config.dasd_recovery_forced_leaf_budget)),
            max_branch_width=max(1, min(self._fixed_max_branch_width, config.dasd_recovery_forced_leaf_budget)),
        )

    def _is_dasd_unstable(self, state: DasdRequestState, base_token_index: Optional[int] = None):
        base = state.committed_len if base_token_index is None else base_token_index
        return (
            state.local_stabilization_active
            or state.cooldown_active
            or state.recovery_mode_active
            or state.consecutive_full_rejections > 0
            or state.base_retry_counts.get(base, 0) > 1
        )

    def _current_dasd_inflight_cap(self, state: DasdRequestState):
        if state.frontier_sync_active:
            state.frontier_sync_inflight_cap_hits += 1
            self._log_dasd_state_event(
                "frontier_sync_inflight_cap_applied",
                state,
                decision_reason=state.last_frontier_sync_reason or "frontier_sync_active",
                base_token_index=state.committed_len,
                target_committed_len=state.frontier_sync_target_committed_len,
                remaining_progress=state.frontier_sync_progress_remaining,
                inflight_cap=1,
            )
            return 1
        if self._is_dasd_unstable(state):
            state.unstable_phase_inflight_cap_hits += 1
            self._log_dasd_state_event(
                "unstable_inflight_cap_applied",
                state,
                decision_reason="unstable_phase",
                base_token_index=state.committed_len,
                inflight_cap=1,
            )
            return 1
        return config.dasd_max_inflight_bundles

    def _enter_local_stabilization(self, state: DasdRequestState, reason: str):
        if state.local_stabilization_active:
            return
        state.local_stabilization_active = True
        state.local_stabilization_attempt_count += 1
        state.last_mitigation_decision_reason = reason
        self._refresh_dasd_control_targets(
            state,
            reason="local_stabilization_enter",
            next_credit=state.window_size,
        )
        self._log_dasd_state_event(
            "local_stabilization_enter",
            state,
            decision_reason=reason,
            base_token_index=state.committed_len,
        )

    def _exit_local_stabilization(self, state: DasdRequestState, reason: str, success: bool):
        if not state.local_stabilization_active:
            return
        state.local_stabilization_active = False
        state.last_mitigation_decision_reason = reason
        self._refresh_dasd_control_targets(
            state,
            reason="local_stabilization_exit",
            next_credit=state.window_size,
        )
        if success:
            state.local_stabilization_success_count += 1
            self._log_dasd_state_event(
                "local_stabilization_success",
                state,
                decision_reason=reason,
                base_token_index=state.committed_len,
            )
        else:
            state.local_stabilization_fail_count += 1
            self._log_dasd_state_event(
                "local_stabilization_fail",
                state,
                decision_reason=reason,
                base_token_index=state.committed_len,
            )

    def _enter_cooldown(self, state: DasdRequestState, reason: str):
        state.cooldown_active = True
        state.cooldown_progress_remaining = 2
        state.cooldown_entry_count += 1
        state.last_cooldown_reason = reason
        self._refresh_dasd_control_targets(
            state,
            reason="cooldown_enter",
            next_credit=state.window_size,
        )
        self._log_dasd_state_event(
            "cooldown_enter",
            state,
            decision_reason=reason,
            base_token_index=state.committed_len,
        )

    def _record_cooldown_progress(self, state: DasdRequestState, reason: str):
        if not state.cooldown_active:
            return
        state.cooldown_progress_count += 1
        state.cooldown_progress_remaining = max(0, state.cooldown_progress_remaining - 1)
        self._log_dasd_state_event(
            "cooldown_progress",
            state,
            decision_reason=reason,
            base_token_index=state.committed_len,
            remaining=state.cooldown_progress_remaining,
        )
        if state.cooldown_progress_remaining == 0:
            state.cooldown_active = False
            state.cooldown_exit_count += 1
            state.last_cooldown_reason = reason
            self._refresh_dasd_control_targets(
                state,
                reason="cooldown_exit",
                next_credit=state.window_size,
            )
            self._log_dasd_state_event(
                "cooldown_exit",
                state,
                decision_reason=reason,
                base_token_index=state.committed_len,
            )

    def _enter_frontier_sync(self, state: DasdRequestState, reason: str):
        if state.frontier_sync_active:
            state.frontier_sync_target_committed_len = state.committed_len
            state.frontier_sync_progress_remaining = max(
                state.frontier_sync_progress_remaining,
                2,
            )
            state.last_frontier_sync_reason = reason
            self._refresh_dasd_control_targets(
                state,
                reason="frontier_sync_refresh",
                next_credit=state.window_size,
            )
            self._log_dasd_state_event(
                "frontier_sync_enter",
                state,
                decision_reason=reason,
                base_token_index=state.committed_len,
                target_committed_len=state.frontier_sync_target_committed_len,
                remaining_progress=state.frontier_sync_progress_remaining,
            )
            return
        state.frontier_sync_active = True
        state.frontier_sync_target_committed_len = state.committed_len
        state.frontier_sync_progress_remaining = 2
        state.frontier_sync_entry_count += 1
        state.last_frontier_sync_reason = reason
        self._refresh_dasd_control_targets(
            state,
            reason="frontier_sync_enter",
            next_credit=state.window_size,
        )
        self._log_dasd_state_event(
            "frontier_sync_enter",
            state,
            decision_reason=reason,
            base_token_index=state.committed_len,
            target_committed_len=state.frontier_sync_target_committed_len,
            remaining_progress=state.frontier_sync_progress_remaining,
        )

    def _record_frontier_sync_progress(self, state: DasdRequestState, reason: str):
        if not state.frontier_sync_active:
            return
        target = state.frontier_sync_target_committed_len
        if target is None or state.committed_len <= target:
            return
        state.frontier_sync_progress_remaining = max(
            0, state.frontier_sync_progress_remaining - 1
        )
        state.frontier_sync_target_committed_len = state.committed_len
        self._log_dasd_state_event(
            "frontier_sync_progress",
            state,
            decision_reason=reason,
            base_token_index=state.committed_len,
            target_committed_len=state.frontier_sync_target_committed_len,
            remaining_progress=state.frontier_sync_progress_remaining,
        )
        if state.frontier_sync_progress_remaining == 0:
            self._exit_frontier_sync(state, reason=reason, success=True)

    def _exit_frontier_sync(self, state: DasdRequestState, reason: str, success: bool):
        if not state.frontier_sync_active:
            return
        state.frontier_sync_active = False
        state.frontier_sync_target_committed_len = None
        state.frontier_sync_progress_remaining = 0
        state.frontier_sync_exit_count += 1
        if success:
            state.frontier_sync_success_count += 1
        else:
            state.frontier_sync_fail_count += 1
        state.last_frontier_sync_reason = reason
        self._refresh_dasd_control_targets(
            state,
            reason="frontier_sync_exit",
            next_credit=state.window_size,
        )
        self._log_dasd_state_event(
            "frontier_sync_exit",
            state,
            decision_reason=reason,
            base_token_index=state.committed_len,
            target_committed_len=state.frontier_sync_target_committed_len,
            remaining_progress=state.frontier_sync_progress_remaining,
            success=success,
        )

    def _is_frontier_sync_send_allowed(
        self, state: DasdRequestState, base_token_index: int
    ):
        return (
            not state.frontier_sync_active
            or base_token_index == state.committed_len
        )

    def _record_retry_fingerprint(
        self,
        state: DasdRequestState,
        base_token_index: int,
        token_ids: list[int],
    ):
        lifecycle = self._ensure_base_lifecycle(state, base_token_index)
        fingerprint = self._retry_fingerprint(state, base_token_index, token_ids)
        fingerprints = state.retry_fingerprints_by_base.setdefault(base_token_index, {})
        fingerprints[fingerprint] = fingerprints.get(fingerprint, 0) + 1
        lifecycle["retry_fingerprints"][fingerprint] = fingerprints[fingerprint]
        lifecycle["unique_retry_fingerprints"].add(fingerprint)
        if fingerprints[fingerprint] == 1:
            state.unique_retry_fingerprint_count += 1
        else:
            state.duplicate_retry_fingerprint_count += 1
        if (
            fingerprints[fingerprint] >= 2
            and lifecycle["accepted_tokens_accumulated"] == 0
        ):
            state.retry_loop_suspected_count += 1
            self._log_dasd_state_event(
                "retry_loop_suspected",
                state,
                decision_reason="duplicate_retry_fingerprint",
                base_token_index=base_token_index,
                fingerprint_attempts=fingerprints[fingerprint],
                unique_retry_fingerprints=len(lifecycle["unique_retry_fingerprints"]),
                failure_cache_changed=bool(self._blocked_tokens_for_prefix(state)),
                recovery_mode_active=state.recovery_mode_active,
            )
            self._log_dasd_state_event(
                "retry_loop_breaker_candidate",
                state,
                decision_reason=(
                    "forced_commit_candidate"
                    if state.recovery_mode_active
                    else "recovery_mode_candidate"
                ),
                base_token_index=base_token_index,
                fingerprint_attempts=fingerprints[fingerprint],
                unique_retry_fingerprints=len(lifecycle["unique_retry_fingerprints"]),
            )
        return fingerprints[fingerprint], len(lifecycle["unique_retry_fingerprints"])

    def _record_rollback_cause(
        self,
        state: DasdRequestState,
        cause: str,
        base_token_index: int,
        **extra,
    ):
        state.last_rollback_cause = cause
        count_field = f"{cause}_count"
        if hasattr(state, count_field):
            setattr(state, count_field, getattr(state, count_field) + 1)
        lifecycle = self._ensure_base_lifecycle(state, base_token_index)
        lifecycle["rollback_events"] += 1
        self._log_dasd_state_event(
            "rollback_decision",
            state,
            decision_reason=cause,
            base_token_index=base_token_index,
            **extra,
        )

    def _can_continue_dasd_generation(self, state: DasdRequestState):
        return (
            not state.aborted
            and state.committed_len < self._max_new_tokens
            and not self._dasd_eos_seen(state)
        )

    def _clear_suffix_refresh_anchor(self, state: DasdRequestState):
        state.suffix_refresh_anchor_committed_len = None

    def _prepare_dasd_refill_from_committed(self, state: DasdRequestState, reason: str):
        before_next_base = state.next_base_index
        before_drafted_len = len(state.drafted_tokens)
        before_pending_responses = len(state.responses_by_base)
        if config.dasd_debug:
            self._logger.info(
                "[DASD] rebuild_restart_begin req=%s epoch=%d committed=%d drafted=%d next_base=%d inflight=%d pending=%d reason=%s credit=%s W=%d tree_depth=%s leaf_budget=%s",
                state.request_id,
                state.epoch,
                state.committed_len,
                len(state.drafted_tokens),
                state.next_base_index,
                len(state.inflight),
                before_pending_responses,
                reason,
                state.credit_controller.credit if state.credit_controller is not None else None,
                state.window_size,
                state.tree_budget.max_beam_len if state.tree_budget is not None else None,
                state.tree_budget.max_budget if state.tree_budget is not None else None,
            )
        state.next_base_index = state.committed_len
        state.drafted_tokens = state.drafted_tokens[: state.committed_len]
        state.responses_by_base.clear()
        state.cleanup_induced_drain = False
        if (
            before_next_base != state.next_base_index
            or before_drafted_len != len(state.drafted_tokens)
            or before_pending_responses > 0
        ):
            self._logger.info(
                "[DASD] inconsistency_detected req=%s epoch=%d reason=%s before_next_base=%d after_next_base=%d before_drafted_len=%d after_drafted_len=%d pending_responses_before=%d correction_applied=True",
                state.request_id,
                state.epoch,
                reason,
                before_next_base,
                state.next_base_index,
                before_drafted_len,
                len(state.drafted_tokens),
                before_pending_responses,
            )
        if config.dasd_debug:
            prefix_tail = state.drafted_tokens[max(0, state.committed_len - 8) : state.committed_len]
            blocked_tokens = sorted(self._blocked_tokens_for_prefix(state))
            self._logger.info(
                "[DASD] rollback_rebuild_start req=%s epoch=%d committed=%d next_base=%d reason=%s prefix_tail=%s blocked_tokens=%s",
                state.request_id,
                state.epoch,
                state.committed_len,
                state.next_base_index,
                reason,
                prefix_tail,
                blocked_tokens,
            )
        state.rollback_rebuild_count += 1
        self._reset_tree_from_dasd_state(state, use_full_drafted=False)
        if config.dasd_debug:
            prefix_tail = state.drafted_tokens[max(0, state.committed_len - 8) : state.committed_len]
            self._logger.info(
                "[DASD] rebuild_restart_end req=%s epoch=%d committed=%d drafted_len=%d next_base=%d inflight=%d pending=%d prefix_tail=%s",
                state.request_id,
                state.epoch,
                state.committed_len,
                len(state.drafted_tokens),
                state.next_base_index,
                len(state.inflight),
                len(state.responses_by_base),
                prefix_tail,
            )

    def _prepare_dasd_suffix_refresh_state(self, state: DasdRequestState, reason: str):
        anchor_committed_len = state.committed_len
        if state.suffix_refresh_anchor_committed_len != anchor_committed_len:
            self._log_dasd_state_event(
                "suffix_refresh_anchor_miss",
                state,
                decision_reason=reason,
                base_token_index=state.committed_len,
                anchor_committed_len=state.suffix_refresh_anchor_committed_len,
                current_committed_len=state.committed_len,
                current_epoch=state.epoch,
                truncating_rebuild_applied=True,
            )
            self._prepare_dasd_refill_from_committed(state, reason="suffix_refresh")
            state.suffix_refresh_anchor_committed_len = anchor_committed_len
            return

        self._log_dasd_state_event(
            "suffix_refresh_anchor_hit",
            state,
            decision_reason=reason,
            base_token_index=state.committed_len,
            anchor_committed_len=state.suffix_refresh_anchor_committed_len,
            current_committed_len=state.committed_len,
            current_epoch=state.epoch,
            truncating_rebuild_applied=False,
        )

        before_next_base = state.next_base_index
        before_pending_responses = len(state.responses_by_base)
        if state.next_base_index < state.committed_len:
            state.next_base_index = state.committed_len
        if state.responses_by_base:
            state.responses_by_base.clear()
        if (
            before_next_base != state.next_base_index
            or before_pending_responses > 0
        ):
            self._logger.info(
                "[DASD] inconsistency_detected req=%s epoch=%d reason=%s before_next_base=%d after_next_base=%d drafted_len=%d pending_responses_before=%d correction_applied=True",
                state.request_id,
                state.epoch,
                reason,
                before_next_base,
                state.next_base_index,
                len(state.drafted_tokens),
                before_pending_responses,
            )

    def _regenerate_dasd_suffix(
        self,
        state: DasdRequestState,
        min_required_end: int,
        phase: str,
        max_attempts: int = 3,
    ):
        if len(state.drafted_tokens) >= min_required_end:
            return True, "suffix_already_available"

        if config.dasd_debug:
            self._logger.info(
                "[DASD] draft_regeneration_begin req=%s epoch=%d committed=%d drafted=%d next_base=%d min_required_end=%d inflight=%d phase=%s credit=%s W=%d tree_depth=%s leaf_budget=%s",
                state.request_id,
                state.epoch,
                state.committed_len,
                len(state.drafted_tokens),
                state.next_base_index,
                min_required_end,
                len(state.inflight),
                phase,
                state.credit_controller.credit if state.credit_controller is not None else None,
                state.window_size,
                state.tree_budget.max_beam_len if state.tree_budget is not None else None,
                state.tree_budget.max_budget if state.tree_budget is not None else None,
            )

        last_reason = "draft_regeneration_produced_no_suffix"
        for attempt_idx in range(1, max_attempts + 1):
            state.draft_regeneration_attempt_count += 1
            remaining_capacity = (
                config.dasd_max_spec_buffer_tokens - state.speculative_tokens()
            )
            if remaining_capacity <= 0:
                last_reason = "send_budget_exhausted"
                break
            missing_tokens = max(1, min_required_end - len(state.drafted_tokens))
            target_tokens = min(
                remaining_capacity,
                max(state.window_size, missing_tokens),
            )
            before_len = len(state.drafted_tokens)
            self._draft_more_for_dasd(
                state,
                target_tokens,
                preserve_speculative_suffix_only=(phase == "suffix_refresh"),
            )
            after_len = len(state.drafted_tokens)
            if after_len >= min_required_end:
                state.draft_regeneration_success_count += 1
                if config.dasd_debug:
                    self._logger.info(
                        "[DASD] draft_regeneration_success req=%s epoch=%d committed=%d drafted=%d next_base=%d min_required_end=%d attempt=%d",
                        state.request_id,
                        state.epoch,
                        state.committed_len,
                        after_len,
                        state.next_base_index,
                        min_required_end,
                        attempt_idx,
                    )
                return True, "draft_regeneration_success"
            if after_len <= before_len:
                last_reason = "draft_regeneration_produced_no_suffix"
                state.draft_regeneration_no_suffix_count += 1
                if config.dasd_debug:
                    self._logger.info(
                        "[DASD] draft_regeneration_no_suffix req=%s epoch=%d committed=%d drafted=%d next_base=%d min_required_end=%d attempt=%d",
                        state.request_id,
                        state.epoch,
                        state.committed_len,
                        after_len,
                        state.next_base_index,
                        min_required_end,
                        attempt_idx,
                    )
                break
            last_reason = "draft_regeneration_insufficient_suffix"

        if config.dasd_debug:
            self._logger.info(
                "[DASD] draft_regeneration_failed req=%s epoch=%d committed=%d drafted=%d next_base=%d min_required_end=%d reason=%s",
                state.request_id,
                state.epoch,
                state.committed_len,
                len(state.drafted_tokens),
                state.next_base_index,
                min_required_end,
                last_reason,
            )
        return False, last_reason

    def _perform_suffix_refresh(self, state: DasdRequestState, reason: str):
        state.suffix_refresh_attempt_count += 1
        state.last_suffix_refresh_reason = reason
        state.last_mitigation_decision_reason = reason
        self._refresh_dasd_control_targets(
            state,
            reason="suffix_refresh_begin",
            next_credit=state.window_size,
        )
        self._log_dasd_state_event(
            "suffix_refresh_begin",
            state,
            decision_reason=reason,
            base_token_index=state.committed_len,
        )
        self._prepare_dasd_suffix_refresh_state(state, reason=reason)
        regenerated, regen_reason = self._regenerate_dasd_suffix(
            state,
            min_required_end=state.next_base_index + state.window_size,
            phase="suffix_refresh",
        )
        if regenerated:
            state.suffix_refresh_success_count += 1
            state.rollback_cleanup_avoided_count += 1
            state.cheap_recovery_success_count += 1
            self._enter_frontier_sync(state, reason="suffix_refresh_success")
            self._log_dasd_state_event(
                "suffix_refresh_end",
                state,
                decision_reason="success",
                base_token_index=state.committed_len,
            )
            self._log_dasd_state_event(
                "suffix_refresh_success",
                state,
                decision_reason=reason,
                base_token_index=state.committed_len,
            )
            self._enter_cooldown(state, reason="suffix_refresh_success")
            return True, "suffix_refresh_success"
        state.suffix_refresh_fail_count += 1
        self._log_dasd_state_event(
            "suffix_refresh_end",
            state,
            decision_reason=regen_reason,
            base_token_index=state.committed_len,
        )
        self._log_dasd_state_event(
            "suffix_refresh_fail",
            state,
            decision_reason=regen_reason,
            base_token_index=state.committed_len,
        )
        return False, regen_reason

    def _should_short_circuit_same_base_retry(
        self,
        state: DasdRequestState,
        base_token_index: int,
        fingerprint_attempts: int,
        unique_fingerprints: int,
    ):
        if base_token_index != state.committed_len:
            return False, ""
        if state.base_retry_counts.get(base_token_index, 0) <= 1:
            return False, ""
        blocked_tokens = self._blocked_tokens_for_prefix(state)
        if fingerprint_attempts >= 2 and state.consecutive_full_rejections > 0:
            return True, "duplicate_retry_fingerprint"
        if (
            self._is_dasd_unstable(state, base_token_index=base_token_index)
            and unique_fingerprints <= 2
            and len(blocked_tokens) > 0
            and state.base_retry_counts.get(base_token_index, 0) >= 2
        ):
            return True, "low_value_same_base_retry"
        return False, ""

    def _should_defer_rollback_cleanup(
        self,
        state: DasdRequestState,
        base_token_index: int,
        rollback_cause: str,
    ):
        if rollback_cause == "rollback_due_to_state_inconsistency":
            return False
        if rollback_cause == "rollback_due_to_contiguous_commit_mismatch":
            return False
        retry_count = state.base_retry_counts.get(base_token_index, 0)
        defer_retry_limit = max(2, config.dasd_recovery_same_base_retry_threshold)
        return (
            state.local_stabilization_active
            or state.recovery_mode_active
            or retry_count <= defer_retry_limit
        )

    async def _attempt_dasd_refill_recovery(
        self,
        state: DasdRequestState,
        reason: str,
        bounded_attempts: int = 1,
    ):
        if not self._can_continue_dasd_generation(state):
            self._log_finish_condition(state, reason=f"refill_skipped:{reason}")
            return False

        state.expensive_recovery_count += 1

        if reason == "unexpected_empty_inflight":
            state.unexpected_empty_inflight_recovery_count += 1
            if config.dasd_debug:
                self._logger.info(
                    "[DASD] unexpected_empty_inflight_recovery_start req=%s epoch=%d committed=%d next_base=%d drafted_len=%d inflight=%d",
                    state.request_id,
                    state.epoch,
                    state.committed_len,
                    state.next_base_index,
                    len(state.drafted_tokens),
                    len(state.inflight),
                )

        for attempt_idx in range(1, bounded_attempts + 1):
            state.refill_attempt_count += 1
            self._log_refill_decision(
                state,
                phase=reason,
                action="attempt",
                reason=reason,
                attempt=attempt_idx,
            )
            if state.frontier_sync_active:
                state.next_base_index = state.committed_len
            self._prepare_dasd_refill_from_committed(state, reason=reason)
            regenerated, regeneration_reason = self._regenerate_dasd_suffix(
                state,
                min_required_end=state.next_base_index + state.window_size,
                phase=reason,
            )
            if not regenerated:
                self._log_refill_decision(
                    state,
                    phase=reason,
                    action="fail",
                    reason=regeneration_reason,
                    attempt=attempt_idx,
                )
                break

            if config.dasd_debug:
                self._logger.info(
                    "[DASD] bundle_build_after_regeneration req=%s epoch=%d committed=%d drafted=%d next_base=%d inflight=%d phase=%s",
                    state.request_id,
                    state.epoch,
                    state.committed_len,
                    len(state.drafted_tokens),
                    state.next_base_index,
                    len(state.inflight),
                    reason,
                )

            refill_result = await self._fill_inflight_for_dasd(
                state,
                phase=reason,
                allow_regeneration=False,
            )
            if state.inflight:
                state.refill_success_count += 1
                state.recovery_resume_success_count += 1
                self._enter_frontier_sync(state, reason="rebuild_recovery_success")
                state.frontier_sync_resume_count += 1
                self._log_dasd_state_event(
                    "frontier_sync_resume",
                    state,
                    decision_reason="rebuild_recovery_success",
                    base_token_index=state.committed_len,
                    target_committed_len=state.frontier_sync_target_committed_len,
                    remaining_progress=state.frontier_sync_progress_remaining,
                )
                self._exit_local_stabilization(state, reason="rebuild_recovery_success", success=True)
                self._enter_cooldown(state, reason="rebuild_recovery_success")
                if config.dasd_debug:
                    self._logger.info(
                        "[DASD] recovery_resume_success req=%s epoch=%d committed=%d drafted=%d next_base=%d inflight=%d phase=%s",
                        state.request_id,
                        state.epoch,
                        state.committed_len,
                        len(state.drafted_tokens),
                        state.next_base_index,
                        len(state.inflight),
                        reason,
                    )
                if reason == "unexpected_empty_inflight":
                    state.premature_stall_prevented_count += 1
                self._log_refill_decision(
                    state,
                    phase=reason,
                    action="success",
                    reason="send_spawned",
                    attempt=attempt_idx,
                    send_spawn_count=refill_result.get("send_spawn_count", 0),
                )
                return True

        state.refill_skip_count += 1
        state.recovery_resume_fail_count += 1
        self._exit_local_stabilization(state, reason="rebuild_recovery_failed", success=False)
        if config.dasd_debug:
            self._logger.info(
                "[DASD] recovery_resume_fail req=%s epoch=%d committed=%d drafted=%d next_base=%d inflight=%d phase=%s reason=%s",
                state.request_id,
                state.epoch,
                state.committed_len,
                len(state.drafted_tokens),
                state.next_base_index,
                len(state.inflight),
                reason,
                state.last_recovery_failure_reason or state.last_refill_skip_reason,
            )
        if reason == "unexpected_empty_inflight":
            state.unexpected_empty_inflight_recovery_fail_count += 1
        self._log_refill_decision(
            state,
            phase=reason,
            action="fail",
            reason=state.last_refill_skip_reason or "recovery_attempt_limit_reached",
            attempts=bounded_attempts,
        )
        return False

    def _cancel_dasd_inflight(self, state: DasdRequestState, reason: str):
        task_infos = list(state.inflight.values())
        before_count = len(task_infos)
        if before_count == 0:
            return []

        state.inflight_cleanup_count += 1
        state.cleanup_reason = reason
        state.cleanup_induced_drain = True
        state.inflight.clear()
        state.task_to_bundle.clear()
        state.responses_by_base.clear()

        for task_info in task_infos:
            task_info.task.cancel()

        if config.dasd_debug:
            self._logger.info(
                "[DASD] request_cleanup_start req=%s epoch=%d reason=%s inflight_before=%d inflight_after=%d",
                state.request_id,
                state.epoch,
                reason,
                before_count,
                len(state.inflight),
            )
        return task_infos

    async def _await_cancelled_dasd_tasks(
        self,
        state: DasdRequestState,
        task_infos: list[DasdTaskInfo],
        reason: str,
    ):
        if not task_infos:
            return
        results = await asyncio.gather(
            *[task_info.task for task_info in task_infos],
            return_exceptions=True,
        )
        late_drop_count = 0
        for task_info, result in zip(task_infos, results):
            if isinstance(result, asyncio.CancelledError):
                continue
            if isinstance(result, BaseException):
                if isinstance(result, grpc.aio.AioRpcError) and result.code() == grpc.StatusCode.CANCELLED:
                    continue
                late_drop_count += 1
                if config.dasd_debug:
                    self._logger.info(
                        "[DASD] late_task_dropped req=%s epoch=%d bundle=%d base=%d reason=%s outcome=%s",
                        state.request_id,
                        task_info.epoch_at_send,
                        task_info.bundle_id,
                        task_info.base_token_index,
                        reason,
                        type(result).__name__,
                    )
                continue
            late_drop_count += 1
            if config.dasd_debug:
                self._logger.info(
                    "[DASD] late_task_dropped req=%s epoch=%d bundle=%d base=%d reason=%s outcome=response",
                    state.request_id,
                    task_info.epoch_at_send,
                    task_info.bundle_id,
                    task_info.base_token_index,
                    reason,
                )
        state.late_task_drop_count += late_drop_count
        if config.dasd_debug:
            self._logger.info(
                "[DASD] request_cleanup_end req=%s epoch=%d reason=%s late_dropped=%d",
                state.request_id,
                state.epoch,
                reason,
                late_drop_count,
            )

    def _verify_configs(self):
        if self._proactive_type not in ["included", "excluded", "disabled"]:
            raise ValueError(f"Invalid proactive_type: {self._proactive_type}")
        if self._mode not in ["specedge", "dasd"]:
            raise ValueError(f"Invalid mode: {self._mode}")

    async def generate(self, req_idx: int):
        """
        Generate a sequence using SpecExec up to max_new_tokens.
        """

        if self._dasd_enabled:
            try:
                await self._generate_dasd(req_idx)
            finally:
                await self._validator.close()
            return

        self._logger.info("Generating sequence req_idx=%d", req_idx)

        util.set_seed(config.seed)
        step_idx = 0

        # Prefill phase
        self._logger.debug("Prefill phase: req_idx=%d, step_idx=%d", req_idx, step_idx)
        warmup_tokens = await self._cycle(req_idx, step_idx, prefill=True)
        self._prefix_tokens = torch.cat([self._prefix_tokens, warmup_tokens], dim=-1)

        step_idx = 1
        eos_flag = False

        # speculative decoding phase
        while (
            self._prefix_tokens.numel()
            < self._max_new_tokens + self._num_original_tokens + warmup_tokens.numel()
            and not eos_flag
        ):
            self._logger.debug(
                "Speculative Decoding phase: req_idx=%d, step_idx=%d", req_idx, step_idx
            )
            fresh_tokens = await self._cycle(req_idx, step_idx)

            eos_positions = (fresh_tokens == self._tokenizer.eos_token_id).nonzero()
            if eos_positions.numel() > 0:
                eos_idx = eos_positions[0, 0].item()
                fresh_tokens = fresh_tokens[: eos_idx + 1]
                eos_flag = True

            self._prefix_tokens = torch.cat([self._prefix_tokens, fresh_tokens], dim=-1)
            step_idx += 1

        if eos_flag:
            self._logger.debug("EOS token found.")
        else:
            self._logger.debug("Max new tokens reached.")

        self._logger.info("Finished generating sequence req_idx=%d", req_idx)
        self._logger.info(
            "Generated sequence: \n%s",
            self._tokenizer.decode(self._prefix_tokens[0], skip_special_tokens=True),
        )
        await self._validator.close()

    async def _cycle(self, req_idx: int, step_idx: int, prefill=False) -> torch.Tensor:
        with util.Timing(device=self._device, mode="sync") as draft_t:
            draft_stats = self._grow_tree(prefill)

        with util.Timing(device=self._device, mode="sync") as target_t:
            fresh_token_ids, target_stats = await self._validate_tree(req_idx, prefill)

        self._result_logger.log(
            {
                "client_idx": self._client_idx,
                "req_idx": req_idx,
                "step_idx": step_idx,
                "draft": {
                    "forward": draft_stats["forward_t"],
                    "end_to_end": draft_t.elapsed,
                },
                "target": {
                    "client_preprocess": target_stats["preprocess_t"],
                    "client_wait": target_stats["wait_t"],
                    "client_postprocess": target_stats["postprocess_t"],
                    "end_to_end": target_t.elapsed,
                    "prefill": target_stats["prefill"],
                    "proactive": target_stats["proactive"],
                    "prev_proactive": target_stats["previous_proactive"],
                },
                "num_accepted_tokens": target_stats["num_accepted_tokens"],
            }
        )

        return fresh_token_ids

    async def _generate_dasd(self, req_idx: int):
        self._logger.info("Generating sequence req_idx=%d (dasd)", req_idx)

        state = DasdRequestState(
            req_idx=req_idx,
            request_id=str(req_idx),
            window_size=max(
                config.dasd_w_min, min(config.dasd_w_max, config.dasd_start_window)
            ),
            credit_controller=DasdCreditController(
                adaptive_enabled=config.dasd_adaptive_credit_enabled,
                adaptive_window_enabled=config.dasd_adaptive_window_enabled,
                adaptive_tree_budget_enabled=config.dasd_adaptive_tree_budget_enabled,
                credit_min=config.dasd_credit_min,
                credit_max=max(config.dasd_credit_min, config.dasd_credit_max),
                credit=max(
                    config.dasd_credit_min,
                    min(config.dasd_credit_max, config.dasd_credit_init),
                ),
                rejection_penalty=config.dasd_rejection_penalty,
                success_bonus=config.dasd_success_bonus,
                min_window=config.dasd_min_window,
                max_window=config.dasd_max_window,
                min_tree_depth=config.dasd_min_tree_depth,
                max_tree_depth=config.dasd_max_tree_depth,
                min_leaf_budget=config.dasd_min_leaf_budget,
                max_leaf_budget=config.dasd_max_leaf_budget,
            ),
        )
        self._refresh_dasd_control_targets(
            state,
            reason="init",
            next_credit=state.window_size,
        )
        start_ts = time.perf_counter()

        initial_target = state.window_size * min(2, config.dasd_max_inflight_bundles)
        self._draft_more_for_dasd(state, initial_target)

        eos_flag = False
        finish_status = ""
        while state.committed_len < self._max_new_tokens and not eos_flag and not state.aborted:
            if state.fallback_burst_active:
                progressed = await self._run_fallback_burst_step(state)
                if state.finish_status == "eos_reached":
                    eos_flag = True
                    break
                if state.aborted or state.finish_status in {
                    "max_new_tokens_reached",
                    "explicit_abort_reason",
                }:
                    break
                if progressed:
                    if (
                        state.committed_len > 0
                        and state.committed_len <= len(state.drafted_tokens)
                        and state.drafted_tokens[state.committed_len - 1]
                        == self._tokenizer.eos_token_id
                    ):
                        eos_flag = True
                        state.finish_status = "eos_reached"
                        break
                    continue

            await self._fill_inflight_for_dasd(state, phase="normal_refill")
            if not state.inflight:
                if self._can_continue_dasd_generation(state):
                    refill_reason = (
                        state.cleanup_reason
                        if state.cleanup_induced_drain and state.cleanup_reason
                        else "unexpected_empty_inflight"
                    )
                    if config.dasd_debug:
                        self._logger.info(
                            "[DASD] cleanup_drain req=%s epoch=%d reason=%s finish_status=%s aborted=%s",
                            state.request_id,
                            state.epoch,
                            refill_reason,
                            state.finish_status,
                            state.aborted,
                        )
                    recovered = await self._attempt_dasd_refill_recovery(
                        state,
                        reason=refill_reason,
                    )
                    if recovered:
                        continue
                    self._logger.warning(
                        "[DASD] recovery_failed req=%s epoch=%d reason=%s last_refill_skip_reason=%s last_recovery_failure_reason=%s refill_attempts=%d send_spawn_count=%d send_spawn_after_rebuild_count=%d",
                        state.request_id,
                        state.epoch,
                        refill_reason,
                        state.last_refill_skip_reason,
                        state.last_recovery_failure_reason,
                        state.refill_attempt_count,
                        state.send_spawn_count,
                        state.send_spawn_after_rebuild_count,
                    )
                    state.unexpected_stall_count += 1
                    await self._abort_dasd_request(
                        state,
                        reason="unexpected_empty_inflight_recovery_failed",
                    )
                    finish_status = "explicit_abort_reason"
                else:
                    self._log_finish_condition(state, reason="loop_exit_no_inflight")
                break

            await self._receive_one_round_for_dasd(state)

            if state.committed_len > len(state.drafted_tokens):
                self._logger.warning(
                    "DASD committed_len exceeded drafted buffer (req_idx=%d, committed=%d, drafted=%d). Clamping.",
                    req_idx,
                    state.committed_len,
                    len(state.drafted_tokens),
                )
                state.committed_len = len(state.drafted_tokens)

            if (
                self._dasd_eos_seen(state)
            ):
                eos_flag = True

        if state.finish_status:
            finish_status = state.finish_status
        elif state.aborted:
            finish_status = "explicit_abort_reason"
        elif eos_flag:
            finish_status = "eos_reached"
        elif state.committed_len >= self._max_new_tokens:
            finish_status = "max_new_tokens_reached"
        elif finish_status == "":
            finish_status = "refill_not_possible"
        state.finish_status = finish_status
        self._log_finish_condition(state, reason=finish_status)
        self._exit_frontier_sync(
            state,
            reason=finish_status,
            success=finish_status in {"eos_reached", "max_new_tokens_reached"},
        )

        await self._drain_dasd_inflight(state)
        self._maybe_emit_base_lifecycle_summary(
            state,
            state.current_base_tracking,
            final=True,
            reason=finish_status,
        )

        final_generated = state.drafted_tokens[: state.committed_len]
        if final_generated:
            generated_tokens = torch.tensor(
                final_generated,
                dtype=torch.long,
                device=self._device,
            ).unsqueeze(0)
            self._prefix_tokens = torch.cat(
                [self._initial_prefix_tokens, generated_tokens], dim=-1
            )
        else:
            self._prefix_tokens = self._initial_prefix_tokens.clone()

        total_latency_ms = (time.perf_counter() - start_ts) * 1000.0
        goodput = (
            state.total_accepted_tokens / state.total_verified_tokens
            if state.total_verified_tokens > 0
            else 0.0
        )

        self._result_logger.log(
            {
                "type": "dasd_request_summary",
                "req_idx": req_idx,
                "request_id": state.request_id,
                "prompt_length": self._num_original_tokens,
                "output_length": state.committed_len,
                "total_verify_rounds": state.verify_rounds,
                "total_proposed_tokens": state.total_verified_tokens,
                "total_verified_tokens": state.total_verified_tokens,
                "total_accepted_tokens": state.total_accepted_tokens,
                "acceptance_rate": goodput,
                "goodput": goodput,
                "rollbacks_count": state.rollbacks_count,
                "full_rejection_count": state.full_rejection_count,
                "max_full_rejection_streak": state.max_full_rejection_streak,
                "stall_rounds": state.stall_rounds,
                "same_base_retry_count": state.same_base_retry_count,
                "recovery_mode_entries": state.recovery_mode_entries,
                "failure_cache_hits": state.failure_cache_hits,
                "forced_commit_count": state.forced_commit_count,
                "invalid_token_abort_count": state.invalid_token_abort_count,
                "stale_epoch_drop_count": state.stale_epoch_drop_count,
                "late_task_drop_count": state.late_task_drop_count,
                "inflight_cleanup_count": state.inflight_cleanup_count,
                "rpc_failure_count": state.rpc_failure_count,
                "rpc_unavailable_abort_count": state.rpc_unavailable_abort_count,
                "unexpected_stall_count": state.unexpected_stall_count,
                "refill_attempt_count": state.refill_attempt_count,
                "refill_success_count": state.refill_success_count,
                "refill_skip_count": state.refill_skip_count,
                "rollback_rebuild_count": state.rollback_rebuild_count,
                "unexpected_empty_inflight_recovery_count": state.unexpected_empty_inflight_recovery_count,
                "unexpected_empty_inflight_recovery_fail_count": state.unexpected_empty_inflight_recovery_fail_count,
                "premature_stall_prevented_count": state.premature_stall_prevented_count,
                "last_refill_skip_reason": state.last_refill_skip_reason,
                "last_recovery_failure_reason": state.last_recovery_failure_reason,
                "last_refill_phase": state.last_refill_phase,
                "refill_no_work_count": state.refill_no_work_count,
                "refill_guard_block_count": state.refill_guard_block_count,
                "bundle_build_none_count": state.bundle_build_none_count,
                "send_spawn_count": state.send_spawn_count,
                "send_spawn_after_rebuild_count": state.send_spawn_after_rebuild_count,
                "draft_regeneration_attempt_count": state.draft_regeneration_attempt_count,
                "draft_regeneration_success_count": state.draft_regeneration_success_count,
                "draft_regeneration_no_suffix_count": state.draft_regeneration_no_suffix_count,
                "recovery_resume_success_count": state.recovery_resume_success_count,
                "recovery_resume_fail_count": state.recovery_resume_fail_count,
                "rollback_due_to_full_rejection_threshold_count": state.rollback_due_to_full_rejection_threshold_count,
                "rollback_due_to_same_base_retry_threshold_count": state.rollback_due_to_same_base_retry_threshold_count,
                "rollback_due_to_failure_cache_blocked_suffix_count": state.rollback_due_to_failure_cache_blocked_suffix_count,
                "rollback_due_to_forced_commit_failure_count": state.rollback_due_to_forced_commit_failure_count,
                "rollback_due_to_recovery_mode_entry_count": state.rollback_due_to_recovery_mode_entry_count,
                "rollback_due_to_post_recovery_rejection_count": state.rollback_due_to_post_recovery_rejection_count,
                "rollback_due_to_contiguous_commit_mismatch_count": state.rollback_due_to_contiguous_commit_mismatch_count,
                "rollback_due_to_state_inconsistency_count": state.rollback_due_to_state_inconsistency_count,
                "rollback_due_to_retry_loop_breaker_count": state.rollback_due_to_retry_loop_breaker_count,
                "retry_loop_suspected_count": state.retry_loop_suspected_count,
                "duplicate_retry_fingerprint_count": state.duplicate_retry_fingerprint_count,
                "unique_retry_fingerprint_count": state.unique_retry_fingerprint_count,
                "per_base_max_retry_count": state.per_base_max_retry_count,
                "per_base_max_full_rejection_count": state.per_base_max_full_rejection_count,
                "per_base_max_same_base_retry_count": state.per_base_max_same_base_retry_count,
                "recovery_mode_entry_count": state.recovery_mode_entry_count,
                "recovery_mode_exit_count": state.recovery_mode_exit_count,
                "recovery_mode_success_count": state.recovery_mode_success_count,
                "recovery_mode_failure_count": state.recovery_mode_failure_count,
                "forced_commit_attempt_count": state.forced_commit_attempt_count,
                "forced_commit_success_count": state.forced_commit_success_count,
                "forced_commit_failure_count": state.forced_commit_failure_count,
                "failure_cache_block_decision_count": state.failure_cache_block_decision_count,
                "failure_cache_blocked_same_base_retry_count": state.failure_cache_blocked_same_base_retry_count,
                "local_stabilization_active": state.local_stabilization_active,
                "local_stabilization_attempt_count": state.local_stabilization_attempt_count,
                "local_stabilization_success_count": state.local_stabilization_success_count,
                "local_stabilization_fail_count": state.local_stabilization_fail_count,
                "suffix_refresh_attempt_count": state.suffix_refresh_attempt_count,
                "suffix_refresh_success_count": state.suffix_refresh_success_count,
                "suffix_refresh_fail_count": state.suffix_refresh_fail_count,
                "rollback_cleanup_deferred_count": state.rollback_cleanup_deferred_count,
                "rollback_cleanup_escalated_count": state.rollback_cleanup_escalated_count,
                "rollback_cleanup_avoided_count": state.rollback_cleanup_avoided_count,
                "frontier_sync_active": state.frontier_sync_active,
                "frontier_sync_entry_count": state.frontier_sync_entry_count,
                "frontier_sync_exit_count": state.frontier_sync_exit_count,
                "frontier_sync_success_count": state.frontier_sync_success_count,
                "frontier_sync_fail_count": state.frontier_sync_fail_count,
                "frontier_sync_send_gate_block_count": state.frontier_sync_send_gate_block_count,
                "frontier_sync_inflight_cap_hits": state.frontier_sync_inflight_cap_hits,
                "frontier_sync_resume_count": state.frontier_sync_resume_count,
                "cooldown_active": state.cooldown_active,
                "cooldown_entry_count": state.cooldown_entry_count,
                "cooldown_exit_count": state.cooldown_exit_count,
                "cooldown_progress_count": state.cooldown_progress_count,
                "unstable_phase_inflight_cap_hits": state.unstable_phase_inflight_cap_hits,
                "low_value_retry_suppressed_count": state.low_value_retry_suppressed_count,
                "same_base_retry_short_circuit_count": state.same_base_retry_short_circuit_count,
                "cheap_recovery_success_count": state.cheap_recovery_success_count,
                "expensive_recovery_count": state.expensive_recovery_count,
                "last_rollback_cause": state.last_rollback_cause,
                "last_retry_decision_reason": state.last_retry_decision_reason,
                "last_recovery_mode_transition_reason": state.last_recovery_mode_transition_reason,
                "last_forced_commit_decision_reason": state.last_forced_commit_decision_reason,
                "last_mitigation_decision_reason": state.last_mitigation_decision_reason,
                "last_suffix_refresh_reason": state.last_suffix_refresh_reason,
                "last_cooldown_reason": state.last_cooldown_reason,
                "last_retry_quality_reason": state.last_retry_quality_reason,
                "last_frontier_sync_reason": state.last_frontier_sync_reason,
                "average_W": (
                    state.sum_window_at_send / state.verify_rounds
                    if state.verify_rounds > 0
                    else 0.0
                ),
                "average_tree_depth": (
                    state.sum_tree_depth_at_send / state.verify_rounds
                    if state.verify_rounds > 0
                    else 0.0
                ),
                "average_leaf_budget": (
                    state.sum_leaf_budget_at_send / state.verify_rounds
                    if state.verify_rounds > 0
                    else 0.0
                ),
                "total_latency_ms": total_latency_ms,
                "aborted": state.aborted,
                "abort_reason": state.abort_reason,
                "final_credit": (
                    state.credit_controller.credit
                    if state.credit_controller is not None
                    else None
                ),
                "final_status": state.finish_status,
                "adaptive_window_enabled": config.dasd_adaptive_window_enabled,
                "adaptive_tree_budget_enabled": config.dasd_adaptive_tree_budget_enabled,
                "failure_cache_enabled": config.dasd_failure_cache_enabled,
                "recovery_mode_enabled": config.dasd_recovery_mode_enabled,
                "mode": "dasd",
            }
        )

        self._logger.info("Finished DASD generation req_idx=%d", req_idx)
        self._logger.info(
            "[DASD] generation_finish_reason req=%s reason=%s committed=%d aborted=%s",
            state.request_id,
            state.finish_status,
            state.committed_len,
            state.aborted,
        )
        self._logger.info(
            "Generated sequence: \n%s",
            self._tokenizer.decode(self._prefix_tokens[0], skip_special_tokens=True),
        )

    def _refresh_dasd_control_targets(
        self,
        state: DasdRequestState,
        reason: str,
        next_credit: Optional[int] = None,
        feedback: Optional[dict] = None,
    ):
        if state.credit_controller is None:
            state.tree_budget = DasdTreeBudget(
                max_beam_len=self._fixed_max_beam_len,
                max_budget=self._fixed_max_budget,
                max_n_beams=self._fixed_max_n_beams,
                max_branch_width=self._fixed_max_branch_width,
            )
            return

        previous_window = state.window_size
        previous_tree_budget = state.tree_budget

        if (
            config.dasd_adaptive_credit_enabled
            and config.dasd_adaptive_window_enabled
        ):
            # Joint controller path: credit drives the next external speculation
            # window. The controller bounds come from min/max_window, which
            # already fall back to legacy W_min/W_max in config loading.
            fallback_window = (
                previous_window
                if previous_window > 0
                else max(config.dasd_w_min, min(config.dasd_w_max, config.dasd_start_window))
            )
            state.window_size = state.credit_controller.current_window(fallback_window)
        elif next_credit is not None and next_credit > 0:
            state.window_size = max(
                config.dasd_w_min, min(config.dasd_w_max, int(next_credit))
            )

        state.tree_budget = state.credit_controller.current_tree_budget(
            fallback_depth=self._fixed_max_beam_len,
            fallback_leaf_budget=self._fixed_max_budget,
            fallback_max_n_beams=self._fixed_max_n_beams,
            fallback_max_branch_width=self._fixed_max_branch_width,
        )

        if state.frontier_sync_active:
            conservative_budget = self._conservative_dasd_tree_budget()
            state.window_size = max(1, min(state.window_size, config.dasd_recovery_forced_w))
            state.tree_budget = conservative_budget
            self._log_dasd_state_event(
                "mitigation_decision",
                state,
                decision_reason="frontier_sync_active",
                base_token_index=state.committed_len,
                target_committed_len=state.frontier_sync_target_committed_len,
                remaining_progress=state.frontier_sync_progress_remaining,
            )

        if state.local_stabilization_active or state.cooldown_active:
            conservative_budget = self._conservative_dasd_tree_budget()
            state.window_size = max(1, min(state.window_size, config.dasd_recovery_forced_w))
            state.tree_budget = conservative_budget
            self._log_dasd_state_event(
                "mitigation_decision",
                state,
                decision_reason=(
                    "local_stabilization_active"
                    if state.local_stabilization_active
                    else "cooldown_active"
                ),
                base_token_index=state.committed_len,
            )

        if state.recovery_mode_active and config.dasd_recovery_mode_enabled:
            state.window_size = max(1, config.dasd_recovery_forced_w)
            state.tree_budget = self._conservative_dasd_tree_budget()
            if config.dasd_debug:
                self._logger.info(
                    "[DASD] recovery_mode_active req=%s epoch=%d committed=%d reason=%s rounds_left=%d forced_W=%d forced_tree=%s",
                    state.request_id,
                    state.epoch,
                    state.committed_len,
                    state.recovery_mode_reason,
                    state.recovery_mode_rounds_left,
                    state.window_size,
                    state.tree_budget,
                )

        if feedback is not None:
            feedback["window_mapping"] = {
                "adaptive_window_enabled": (
                    config.dasd_adaptive_credit_enabled
                    and config.dasd_adaptive_window_enabled
                ),
                "credit": state.credit_controller.credit,
                "min_window": config.dasd_min_window,
                "max_window": config.dasd_max_window,
                "chosen_window": state.window_size,
                "server_next_credit": next_credit,
            }
            feedback["tree_mapping"] = {
                "adaptive_tree_budget_enabled": (
                    config.dasd_adaptive_credit_enabled
                    and config.dasd_adaptive_tree_budget_enabled
                ),
                "credit": state.credit_controller.credit,
                "min_tree_depth": config.dasd_min_tree_depth,
                "max_tree_depth": config.dasd_max_tree_depth,
                "min_leaf_budget": config.dasd_min_leaf_budget,
                "max_leaf_budget": config.dasd_max_leaf_budget,
                "chosen_tree_budget": state.tree_budget,
            }

        if config.dasd_debug:
            self._logger.info(
                "[DASD] credit_control req=%s epoch=%d reason=%s adaptive=%s credit_before=%s raw_delta=%s unclamped=%s clamped=%s credit_after=%s accepted=%s proposed=%s rejected=%s strong_accept_streak=%s->%s full_rejection_streak=%s->%s W_before=%s W_after=%s window_mapping=%s tree_before=%s tree_after=%s tree_mapping=%s",
                state.request_id,
                state.epoch,
                reason,
                feedback.get("adaptive_enabled") if feedback is not None else None,
                feedback.get("credit_before") if feedback is not None else state.credit_controller.credit,
                feedback.get("raw_delta") if feedback is not None else None,
                feedback.get("unclamped_credit") if feedback is not None else None,
                feedback.get("clamped_credit") if feedback is not None else None,
                feedback.get("credit_after") if feedback is not None else state.credit_controller.credit,
                feedback.get("accepted_len") if feedback is not None else None,
                feedback.get("proposed_len") if feedback is not None else None,
                feedback.get("rejected_len") if feedback is not None else None,
                feedback.get("strong_accept_streak_before") if feedback is not None else None,
                feedback.get("strong_accept_streak") if feedback is not None else None,
                feedback.get("full_rejection_streak_before") if feedback is not None else None,
                feedback.get("full_rejection_streak") if feedback is not None else None,
                previous_window,
                state.window_size,
                feedback.get("window_mapping") if feedback is not None else None,
                previous_tree_budget,
                state.tree_budget,
                feedback.get("tree_mapping") if feedback is not None else None,
            )

    @contextmanager
    def _dasd_tree_budget_context(self, state: DasdRequestState):
        previous = (
            self._max_beam_len,
            self._max_budget,
            self._max_n_beams,
            self._max_branch_width,
        )
        tree_budget = state.tree_budget or DasdTreeBudget(
            max_beam_len=self._fixed_max_beam_len,
            max_budget=self._fixed_max_budget,
            max_n_beams=self._fixed_max_n_beams,
            max_branch_width=self._fixed_max_branch_width,
        )
        self._max_beam_len = tree_budget.max_beam_len
        self._max_budget = tree_budget.max_budget
        self._max_n_beams = tree_budget.max_n_beams
        self._max_branch_width = tree_budget.max_branch_width
        try:
            yield tree_budget
        finally:
            (
                self._max_beam_len,
                self._max_budget,
                self._max_n_beams,
                self._max_branch_width,
            ) = previous

    def _dasd_prefix_key(self, state: DasdRequestState):
        prefix_tail = tuple(
            state.drafted_tokens[max(0, state.committed_len - 8) : state.committed_len]
        )
        return (state.committed_len, prefix_tail)

    def _purge_expired_failure_cache(self, state: DasdRequestState):
        if not config.dasd_failure_cache_enabled:
            return

        current_round = state.verify_rounds
        expired_prefix_keys = []
        for prefix_key, token_entries in state.failure_cache.items():
            expired_tokens = [
                token_id
                for token_id, entry in token_entries.items()
                if entry.blocked_until_round <= current_round
            ]
            for token_id in expired_tokens:
                entry = token_entries.pop(token_id)
                if config.dasd_debug:
                    self._logger.info(
                        "[DASD] failure_cache_expire req=%s epoch=%d committed=%d base=%d token_id=%s blocked_until_round=%d current_round=%d",
                        state.request_id,
                        state.epoch,
                        prefix_key[0],
                        prefix_key[0],
                        token_id,
                        entry.blocked_until_round,
                        current_round,
                    )
            if not token_entries:
                expired_prefix_keys.append(prefix_key)

        for prefix_key in expired_prefix_keys:
            state.failure_cache.pop(prefix_key, None)

    def _compute_blocked_tokens_for_prefix(self, state: DasdRequestState):
        if not config.dasd_failure_cache_enabled:
            return set()

        self._purge_expired_failure_cache(state)
        prefix_key = self._dasd_prefix_key(state)
        token_entries = state.failure_cache.get(prefix_key, {})
        return {
            token_id
            for token_id, entry in token_entries.items()
            if entry.blocked_until_round > state.verify_rounds
        }

    def _blocked_tokens_for_prefix(self, state: DasdRequestState):
        blocked_tokens = self._compute_blocked_tokens_for_prefix(state)
        if blocked_tokens:
            state.failure_cache_hits += 1
            state.failure_cache_block_decision_count += 1
            if state.base_retry_counts.get(state.committed_len, 0) > 1:
                state.failure_cache_blocked_same_base_retry_count += 1
            if config.dasd_debug:
                self._logger.info(
                    "[DASD] failure_cache_hit req=%s epoch=%d committed=%d base=%d blocked_first_tokens=%s current_round=%d",
                    state.request_id,
                    state.epoch,
                    state.committed_len,
                    state.committed_len,
                    sorted(blocked_tokens),
                    state.verify_rounds,
                )
            self._log_dasd_state_event(
                "failure_cache_decision",
                state,
                decision_reason="blocked_tokens_present",
                base_token_index=state.committed_len,
                blocked_tokens=sorted(blocked_tokens),
                current_round=state.verify_rounds,
            )
        return blocked_tokens

    def _add_failure_cache_token(self, state: DasdRequestState, token_id: Optional[int]):
        if not config.dasd_failure_cache_enabled or token_id is None:
            return

        prefix_key = self._dasd_prefix_key(state)
        token_entries = state.failure_cache.setdefault(prefix_key, {})
        current_round = state.verify_rounds
        entry = token_entries.get(token_id)
        blocked_until_round = current_round + max(1, config.dasd_failure_cache_cooldown)
        if entry is None:
            token_entries[token_id] = DasdFailureCacheEntry(
                token_id=token_id,
                blocked_until_round=blocked_until_round,
                hits=1,
            )
        else:
            entry.hits += 1
            entry.blocked_until_round = blocked_until_round

        max_tokens = max(1, config.dasd_failure_cache_max_tokens_per_prefix)
        if len(token_entries) > max_tokens:
            evict_token_id = min(
                token_entries.items(),
                key=lambda item: (item[1].blocked_until_round, item[1].hits),
            )[0]
            token_entries.pop(evict_token_id, None)

        if config.dasd_debug:
            self._logger.info(
                "[DASD] failure_cache_add req=%s epoch=%d committed=%d base=%d rejected_token_id=%s blocked_until_round=%d cached_tokens=%s",
                state.request_id,
                state.epoch,
                state.committed_len,
                state.committed_len,
                token_id,
                blocked_until_round,
                sorted(token_entries.keys()),
            )

    def _enter_recovery_mode(self, state: DasdRequestState, reason: str):
        if not config.dasd_recovery_mode_enabled:
            return
        previous_window = state.window_size
        previous_tree_budget = state.tree_budget
        state.recovery_mode_active = True
        state.recovery_mode_rounds_left = max(1, config.dasd_recovery_mode_rounds)
        state.recovery_mode_reason = reason
        state.recovery_mode_entries += 1
        state.recovery_mode_entry_count += 1
        state.last_recovery_mode_transition_reason = reason
        self._ensure_base_lifecycle(state, state.committed_len)["recovery_mode_attempts"] += 1
        self._refresh_dasd_control_targets(
            state,
            reason="recovery_mode_enter",
            next_credit=state.window_size,
        )
        self._log_dasd_state_event(
            "recovery_mode_transition",
            state,
            decision_reason=reason,
            base_token_index=state.committed_len,
            transition="enter",
        )
        if config.dasd_debug:
            self._logger.info(
                "[DASD] recovery_mode_enter req=%s epoch=%d committed=%d reason=%s full_rejection_streak=%d same_base_retry_count=%d previous_W=%s forced_W=%s previous_tree=%s forced_tree=%s",
                state.request_id,
                state.epoch,
                state.committed_len,
                reason,
                state.consecutive_full_rejections,
                state.base_retry_counts.get(state.committed_len, 0),
                previous_window,
                state.window_size,
                previous_tree_budget,
                state.tree_budget,
            )

    def _enter_fallback_burst(
        self,
        state: DasdRequestState,
        reason: str,
        sync_base: Optional[int] = None,
        sync_token_id: Optional[int] = None,
    ):
        state.fallback_burst_active = True
        state.fallback_burst_total_steps = max(0, config.dasd_recovery_fallback_decode_steps)
        state.fallback_burst_steps_left = state.fallback_burst_total_steps
        state.fallback_burst_sync_base = sync_base
        state.fallback_burst_sync_token_id = sync_token_id
        if config.dasd_debug:
            self._logger.info(
                "[DASD] fallback_burst_enter req=%s epoch=%d committed=%d base=%d steps=%d reason=%s sync_base=%s sync_token_id=%s",
                state.request_id,
                state.epoch,
                state.committed_len,
                state.committed_len,
                state.fallback_burst_steps_left,
                reason,
                sync_base,
                sync_token_id,
            )

    def _exit_fallback_burst(self, state: DasdRequestState, reason: str):
        if not state.fallback_burst_active:
            return
        if config.dasd_debug:
            self._logger.info(
                "[DASD] fallback_burst_exit req=%s epoch=%d committed=%d base=%d remaining_steps=%d reason=%s",
                state.request_id,
                state.epoch,
                state.committed_len,
                state.committed_len,
                state.fallback_burst_steps_left,
                reason,
            )
        state.fallback_burst_active = False
        state.fallback_burst_steps_left = 0
        state.fallback_burst_total_steps = 0
        state.fallback_burst_sync_base = None
        state.fallback_burst_sync_token_id = None

    def _exit_recovery_mode(self, state: DasdRequestState, reason: str):
        if not state.recovery_mode_active:
            return
        previous_window = state.window_size
        previous_tree_budget = state.tree_budget
        state.recovery_mode_active = False
        state.recovery_mode_rounds_left = 0
        state.recovery_mode_reason = ""
        state.recovery_mode_exit_count += 1
        state.last_recovery_mode_transition_reason = reason
        if reason in {"committed_progress", "forced_commit_progress"}:
            state.recovery_mode_success_count += 1
        else:
            state.recovery_mode_failure_count += 1
        self._refresh_dasd_control_targets(
            state,
            reason="recovery_mode_exit",
            next_credit=state.window_size,
        )
        self._log_dasd_state_event(
            "recovery_mode_transition",
            state,
            decision_reason=reason,
            base_token_index=state.committed_len,
            transition="exit",
        )
        if config.dasd_debug:
            self._logger.info(
                "[DASD] recovery_mode_exit req=%s epoch=%d committed=%d reason=%s previous_W=%s resumed_W=%s previous_tree=%s resumed_tree=%s",
                state.request_id,
                state.epoch,
                state.committed_len,
                reason,
                previous_window,
                state.window_size,
                previous_tree_budget,
                state.tree_budget,
            )

    async def _run_fallback_burst_step(self, state: DasdRequestState):
        if not state.fallback_burst_active:
            return False

        if state.committed_len >= self._max_new_tokens:
            state.finish_status = "max_new_tokens_reached"
            self._exit_fallback_burst(state, reason="max_new_tokens_reached")
            return False

        # One-time server-state realignment for the token that was already forced
        # committed locally at the stuck base.
        if (
            state.fallback_burst_sync_base is not None
            and state.fallback_burst_sync_token_id is not None
        ):
            sync_response = await self._validator.verify_bundle(
                client_id=self._dasd_client_id,
                request_id=state.request_id,
                bundle_id=state.next_bundle_id,
                epoch=state.epoch,
                base_token_index=state.fallback_burst_sync_base,
                token_ids=[],
                timestamp_send_ms=int(time.time() * 1000),
                draft_model_id=config.draft_model,
                recovery_fallback_decode=True,
            )
            state.next_bundle_id += 1
            if int(getattr(sync_response, "verifier_next_token_id", -1)) != int(
                state.fallback_burst_sync_token_id
            ):
                reason = "fallback_sync_mismatch"
                if config.dasd_debug:
                    self._logger.info(
                        "[DASD] fallback_burst_abort req=%s epoch=%d committed=%d base=%d reason=%s verifier_next_token_id=%s expected_sync_token=%s",
                        state.request_id,
                        state.epoch,
                        state.committed_len,
                        state.committed_len,
                        reason,
                        int(getattr(sync_response, "verifier_next_token_id", -1)),
                        state.fallback_burst_sync_token_id,
                    )
                if config.dasd_recovery_abort_on_missing_verifier_token:
                    state.aborted = True
                    state.abort_reason = reason
                    state.finish_status = "explicit_abort_reason"
                self._exit_fallback_burst(state, reason=reason)
                return False
            state.fallback_burst_sync_base = None
            state.fallback_burst_sync_token_id = None

        if state.fallback_burst_steps_left <= 0:
            self._exit_fallback_burst(state, reason="burst_steps_completed")
            return False

        step_idx = state.fallback_burst_total_steps - state.fallback_burst_steps_left + 1
        base_token_index = state.committed_len
        if config.dasd_debug:
            self._logger.info(
                "[DASD] fallback_burst_step req=%s epoch=%d committed=%d base=%d step_idx=%d steps_left=%d",
                state.request_id,
                state.epoch,
                state.committed_len,
                base_token_index,
                step_idx,
                state.fallback_burst_steps_left,
            )

        response = await self._validator.verify_bundle(
            client_id=self._dasd_client_id,
            request_id=state.request_id,
            bundle_id=state.next_bundle_id,
            epoch=state.epoch,
            base_token_index=base_token_index,
            token_ids=[],
            timestamp_send_ms=int(time.time() * 1000),
            draft_model_id=config.draft_model,
            recovery_fallback_decode=True,
        )
        state.next_bundle_id += 1
        token_id = int(getattr(response, "verifier_next_token_id", -1))
        if token_id < 0:
            reason = "missing_verifier_next_token"
            if config.dasd_debug:
                self._logger.info(
                    "[DASD] fallback_burst_abort req=%s epoch=%d committed=%d base=%d step_idx=%d reason=%s",
                    state.request_id,
                    state.epoch,
                    state.committed_len,
                    base_token_index,
                    step_idx,
                    reason,
                )
            if config.dasd_recovery_abort_on_missing_verifier_token:
                state.aborted = True
                state.abort_reason = reason
                state.finish_status = "explicit_abort_reason"
            self._exit_fallback_burst(state, reason=reason)
            return False
        if not self._validate_dasd_token_ids(
            state,
            [token_id],
            source_path="fallback_burst",
            bundle_id=state.next_bundle_id - 1,
            base_token_index=base_token_index,
        ):
            self._schedule_invalid_token_abort(
                state,
                source_path="fallback_burst",
                bundle_id=state.next_bundle_id - 1,
                base_token_index=base_token_index,
            )
            self._exit_fallback_burst(state, reason="invalid_token_id")
            return False

        old_prefix_key = self._dasd_prefix_key(state)
        state.drafted_tokens = state.drafted_tokens[: state.committed_len]
        state.drafted_tokens.append(token_id)
        state.committed_len += 1
        state.next_base_index = state.committed_len
        self._clear_suffix_refresh_anchor(state)
        state.consecutive_full_rejections = 0
        state.rollback_blocked_committed_len = None
        state.rollback_blocked_token_id = None
        state.failure_cache.pop(old_prefix_key, None)
        state.base_retry_counts.pop(base_token_index, None)
        self._reset_tree_from_dasd_state(state, use_full_drafted=False)
        self._enter_frontier_sync(state, reason="fallback_burst_progress")
        blocked_tokens = sorted(self._blocked_tokens_for_prefix(state))
        if config.dasd_debug:
            prefix_tail = state.drafted_tokens[max(0, state.committed_len - 8) : state.committed_len]
            self._logger.info(
                "[DASD] fallback_burst_commit req=%s epoch=%d committed=%d base=%d step_idx=%d token_id=%d prefix_tail=%s blocked_tokens=%s recovery_mode=%s fallback_burst=%s",
                state.request_id,
                state.epoch,
                state.committed_len,
                base_token_index,
                step_idx,
                token_id,
                prefix_tail,
                blocked_tokens,
                state.recovery_mode_active,
                state.fallback_burst_active,
            )
        state.fallback_burst_steps_left -= 1

        if token_id == self._tokenizer.eos_token_id:
            state.finish_status = "eos_reached"
            self._exit_fallback_burst(state, reason="eos_reached")
            return False
        if state.committed_len >= self._max_new_tokens:
            state.finish_status = "max_new_tokens_reached"
            self._exit_fallback_burst(state, reason="max_new_tokens_reached")
            return False
        if state.fallback_burst_steps_left == 0:
            self._exit_fallback_burst(state, reason="burst_steps_completed")
            return False
        return True

    def _should_force_commit(
        self,
        state: DasdRequestState,
        base_token_index: int,
        response,
    ):
        if not (
            config.dasd_recovery_mode_enabled
            and config.dasd_recovery_forced_commit_enabled
            and state.recovery_mode_active
        ):
            state.last_forced_commit_decision_reason = "recovery_mode_inactive"
            return False
        if not bool(getattr(response, "forced_commit_eligible", False)):
            state.last_forced_commit_decision_reason = "not_eligible"
            return False
        if int(getattr(response, "verifier_next_token_id", 0)) < 0:
            state.last_forced_commit_decision_reason = "missing_verifier_token"
            return False
        if state.forced_commits_by_base.get(base_token_index, 0) >= max(
            1, config.dasd_recovery_forced_commit_max_per_base
        ):
            state.last_forced_commit_decision_reason = "max_per_base_reached"
            return False
        same_base_retries = state.base_retry_counts.get(base_token_index, 0)
        if (
            same_base_retries
            >= config.dasd_recovery_forced_commit_same_base_retry_threshold
        ):
            state.last_forced_commit_decision_reason = "same_base_retry_threshold"
            self._log_dasd_state_event(
                "forced_commit_decision",
                state,
                decision_reason=state.last_forced_commit_decision_reason,
                base_token_index=base_token_index,
                forced_commit_eligible=True,
                verifier_next_token_id=int(getattr(response, "verifier_next_token_id", -1)),
            )
            return True
        if (
            state.consecutive_full_rejections
            >= config.dasd_recovery_forced_commit_full_rejection_threshold
        ):
            state.last_forced_commit_decision_reason = "full_rejection_threshold"
            self._log_dasd_state_event(
                "forced_commit_decision",
                state,
                decision_reason=state.last_forced_commit_decision_reason,
                base_token_index=base_token_index,
                forced_commit_eligible=True,
                verifier_next_token_id=int(getattr(response, "verifier_next_token_id", -1)),
            )
            return True
        state.last_forced_commit_decision_reason = "threshold_not_met"
        return False

    def _apply_forced_commit(
        self,
        state: DasdRequestState,
        task_info: DasdTaskInfo,
        response,
    ):
        base_token_index = task_info.base_token_index
        token_id = int(getattr(response, "verifier_next_token_id", -1))
        if token_id < 0:
            state.forced_commit_failure_count += 1
            state.last_forced_commit_decision_reason = "missing_verifier_token"
            return False
        state.forced_commit_attempt_count += 1
        self._ensure_base_lifecycle(state, base_token_index)["forced_commit_attempts"] += 1
        if not self._validate_dasd_token_ids(
            state,
            [token_id],
            source_path="forced_commit",
            bundle_id=task_info.bundle_id,
            base_token_index=base_token_index,
        ):
            self._schedule_invalid_token_abort(
                state,
                source_path="forced_commit",
                bundle_id=task_info.bundle_id,
                base_token_index=base_token_index,
            )
            state.forced_commit_failure_count += 1
            state.last_forced_commit_decision_reason = "invalid_forced_commit_token"
            return False
        old_prefix_key = self._dasd_prefix_key(state)

        if config.dasd_debug:
            self._logger.info(
                "[DASD] forced_commit_trigger req=%s epoch=%d committed=%d base=%d same_base_retries=%d full_rejection_streak=%d recovery_mode=%s verifier_next_token_id=%d",
                state.request_id,
                state.epoch,
                state.committed_len,
                base_token_index,
                state.base_retry_counts.get(base_token_index, 0),
                state.consecutive_full_rejections,
                state.recovery_mode_active,
                token_id,
            )

        state.forced_commits_by_base[base_token_index] = (
            state.forced_commits_by_base.get(base_token_index, 0) + 1
        )
        state.forced_commit_count += 1
        state.forced_commit_success_count += 1
        state.last_forced_commit_decision_reason = "applied"
        self._ensure_base_lifecycle(state, base_token_index)["forced_commit_successes"] += 1
        state.epoch += 1
        cancelled_task_infos = self._cancel_dasd_inflight(
            state, reason="forced_commit_cleanup"
        )
        if cancelled_task_infos:
            drain_task = asyncio.create_task(
                self._await_cancelled_dasd_tasks(
                    state,
                    cancelled_task_infos,
                    reason="forced_commit_cleanup",
                )
            )
            drain_task.add_done_callback(lambda _: None)
        state.drafted_tokens = state.drafted_tokens[: state.committed_len]
        state.drafted_tokens.append(token_id)
        state.committed_len += 1
        state.next_base_index = state.committed_len
        self._clear_suffix_refresh_anchor(state)
        state.consecutive_full_rejections = 0
        state.rollback_blocked_committed_len = None
        state.rollback_blocked_token_id = None

        state.failure_cache.pop(old_prefix_key, None)
        state.base_retry_counts.pop(base_token_index, None)

        self._reset_tree_from_dasd_state(state, use_full_drafted=False)
        if config.dasd_debug:
            self._logger.info(
                "[DASD] forced_commit_apply req=%s epoch=%d committed=%d base=%d token_id=%d forced_commit_count=%d",
                state.request_id,
                state.epoch,
                state.committed_len,
                base_token_index,
                token_id,
                state.forced_commit_count,
            )
        self._enter_fallback_burst(
            state,
            reason="forced_commit_progress",
            sync_base=base_token_index,
            sync_token_id=token_id,
        )
        self._enter_frontier_sync(state, reason="forced_commit_progress")
        self._exit_local_stabilization(
            state,
            reason="forced_commit_progress",
            success=True,
        )
        self._enter_cooldown(state, reason="forced_commit_success")
        self._exit_recovery_mode(state, reason="forced_commit_progress")
        if config.dasd_debug:
            self._logger.info(
                "[DASD] forced_commit_done req=%s epoch=%d committed=%d base=%d token_id=%d recovery_mode=%s",
                state.request_id,
                state.epoch,
                state.committed_len,
                base_token_index,
                token_id,
                state.recovery_mode_active,
            )
        return True

    async def _fill_inflight_for_dasd(
        self,
        state: DasdRequestState,
        phase: str = "send_loop",
        allow_regeneration: bool = True,
    ):
        if state.aborted:
            self._log_refill_decision(
                state,
                phase=phase,
                action="skipped",
                reason="request_aborted",
            )
            return {"reason": "request_aborted", "send_spawn_count": 0}
        if state.committed_len > len(state.drafted_tokens):
            self._logger.warning(
                "[DASD] state_inconsistency req=%s epoch=%d reason=committed_gt_drafted committed=%d drafted=%d",
                state.request_id,
                state.epoch,
                state.committed_len,
                len(state.drafted_tokens),
            )
            state.rollback_due_to_state_inconsistency_count += 1
            state.last_rollback_cause = "rollback_due_to_state_inconsistency"
            state.drafted_tokens = state.drafted_tokens[: state.committed_len]
            self._log_refill_decision(
                state,
                phase=phase,
                action="guard_blocked",
                reason="committed_len_gt_drafted_corrected",
            )
        if state.next_base_index < state.committed_len:
            self._logger.warning(
                "[DASD] state_inconsistency req=%s epoch=%d reason=next_base_lt_committed next_base=%d committed=%d",
                state.request_id,
                state.epoch,
                state.next_base_index,
                state.committed_len,
            )
            state.rollback_due_to_state_inconsistency_count += 1
            state.last_rollback_cause = "rollback_due_to_state_inconsistency"
            state.next_base_index = state.committed_len
            self._log_refill_decision(
                state,
                phase=phase,
                action="guard_blocked",
                reason="next_base_before_committed_corrected",
            )
        inflight_cap = self._current_dasd_inflight_cap(state)
        if len(state.inflight) >= inflight_cap:
            self._log_refill_decision(
                state,
                phase=phase,
                action="skipped",
                reason="max_inflight_reached",
            )
            return {"reason": "max_inflight_reached", "send_spawn_count": 0}
        send_spawn_count = 0
        last_reason = "unknown_guard_blocked"
        while True:
            inflight_cap = self._current_dasd_inflight_cap(state)
            if len(state.inflight) >= inflight_cap:
                break
            if state.frontier_sync_active and state.next_base_index != state.committed_len:
                before_next_base = state.next_base_index
                state.next_base_index = state.committed_len
                self._log_dasd_state_event(
                    "frontier_sync_resume",
                    state,
                    decision_reason="clamp_next_base_to_committed",
                    base_token_index=state.committed_len,
                    target_committed_len=state.frontier_sync_target_committed_len,
                    remaining_progress=state.frontier_sync_progress_remaining,
                    before_next_base=before_next_base,
                    after_next_base=state.next_base_index,
                )
            required_end = (
                state.committed_len + state.window_size
                if state.frontier_sync_active
                else state.next_base_index + state.window_size
            )
            if required_end > len(state.drafted_tokens):
                if allow_regeneration:
                    regenerated, regeneration_reason = self._regenerate_dasd_suffix(
                        state,
                        min_required_end=required_end,
                        phase=phase,
                    )
                    if not regenerated:
                        state.bundle_build_none_count += 1
                        last_reason = regeneration_reason
                        self._log_refill_decision(
                            state,
                            phase=phase,
                            action="no_work",
                            reason=last_reason,
                        )
                        break
                    required_end = state.next_base_index + state.window_size
                else:
                    state.bundle_build_none_count += 1
                    last_reason = "next_base_past_drafted"
                    self._log_refill_decision(
                        state,
                        phase=phase,
                        action="no_work",
                        reason=last_reason,
                    )
                    break

            sent, send_reason = await self._send_bundle_for_dasd(state, phase=phase)
            last_reason = send_reason
            if sent:
                send_spawn_count += 1
                state.send_spawn_count += 1
                if phase != "send_loop":
                    state.send_spawn_after_rebuild_count += 1
                self._log_refill_decision(
                    state,
                    phase=phase,
                    action="scheduled",
                    reason="send_spawned",
                    send_spawn_count=send_spawn_count,
                )
                if (
                    state.frontier_sync_active
                    and state.cleanup_reason == "rollback_cleanup"
                ):
                    state.frontier_sync_resume_count += 1
                    self._log_dasd_state_event(
                        "frontier_sync_resume",
                        state,
                        decision_reason="rollback_cleanup_resume",
                        base_token_index=state.committed_len,
                        target_committed_len=state.frontier_sync_target_committed_len,
                        remaining_progress=state.frontier_sync_progress_remaining,
                    )
                if state.frontier_sync_active:
                    break
            elif send_reason == "low_value_retry_suppressed":
                state.rollback_cleanup_deferred_count += 1
                state.last_mitigation_decision_reason = send_reason
                self._log_dasd_state_event(
                    "rollback_cleanup_deferred",
                    state,
                    decision_reason=send_reason,
                    base_token_index=state.committed_len,
                )
                if not state.local_stabilization_active:
                    self._enter_local_stabilization(state, reason=send_reason)
                self._log_dasd_state_event(
                    "local_stabilization_retry",
                    state,
                    decision_reason=send_reason,
                    base_token_index=state.committed_len,
                )
                refreshed, refresh_reason = self._perform_suffix_refresh(
                    state,
                    reason=send_reason,
                )
                last_reason = refresh_reason
                if refreshed:
                    continue
                self._log_refill_decision(
                    state,
                    phase=phase,
                    action="no_work",
                    reason=last_reason,
                )
                break
            elif send_reason != "request_aborted":
                self._log_refill_decision(
                    state,
                    phase=phase,
                    action="skipped",
                    reason=send_reason,
                )
                break
        if send_spawn_count == 0:
            return {"reason": last_reason, "send_spawn_count": 0}
        return {"reason": "send_spawned", "send_spawn_count": send_spawn_count}

    async def _send_bundle_for_dasd(self, state: DasdRequestState, phase: str = "send_loop"):
        if state.aborted:
            if config.dasd_debug:
                self._logger.info(
                    "[DASD] send_skipped req=%s epoch=%d reason=request_inactive",
                    state.request_id,
                    state.epoch,
                )
            return False, "request_inactive"
        bundle_id = state.next_bundle_id
        base_token_index = state.next_base_index
        if not self._is_frontier_sync_send_allowed(state, base_token_index):
            state.frontier_sync_send_gate_block_count += 1
            state.last_retry_decision_reason = "frontier_sync_wait_for_committed_base"
            self._log_dasd_state_event(
                "frontier_sync_send_blocked",
                state,
                decision_reason=state.last_retry_decision_reason,
                base_token_index=base_token_index,
                target_committed_len=state.frontier_sync_target_committed_len,
                remaining_progress=state.frontier_sync_progress_remaining,
            )
            return False, "frontier_sync_wait_for_committed_base"
        lifecycle = self._ensure_base_lifecycle(state, base_token_index)
        if state.current_base_tracking is None:
            state.current_base_tracking = base_token_index
        elif state.current_base_tracking != base_token_index:
            self._maybe_emit_base_lifecycle_summary(
                state,
                state.current_base_tracking,
                reason="base_transition",
            )
            state.current_base_tracking = base_token_index
        state.base_retry_counts[base_token_index] = (
            state.base_retry_counts.get(base_token_index, 0) + 1
        )
        lifecycle["send_attempts"] += 1
        if state.base_retry_counts[base_token_index] > 1:
            state.same_base_retry_count += 1
            lifecycle["same_base_retry_events"] += 1
            state.per_base_max_same_base_retry_count = max(
                state.per_base_max_same_base_retry_count,
                state.base_retry_counts[base_token_index],
            )
        token_ids = state.drafted_tokens[
            base_token_index : base_token_index + state.window_size
        ]
        if len(token_ids) < state.window_size:
            state.bundle_build_none_count += 1
            return False, "drafted_len_not_ahead_of_committed"
        if not self._validate_dasd_token_ids(
            state,
            token_ids,
            source_path="bundle",
            bundle_id=bundle_id,
            base_token_index=base_token_index,
        ):
            self._schedule_invalid_token_abort(
                state,
                source_path="bundle",
                bundle_id=bundle_id,
                base_token_index=base_token_index,
            )
            return False, "invalid_token_id"
        fingerprint_attempts, unique_fingerprints = self._record_retry_fingerprint(
            state,
            base_token_index=base_token_index,
            token_ids=token_ids,
        )
        state.per_base_max_retry_count = max(
            state.per_base_max_retry_count,
            lifecycle["send_attempts"],
        )
        state.last_retry_decision_reason = (
            "same_base_retry" if state.base_retry_counts[base_token_index] > 1 else "fresh_base_send"
        )
        self._log_dasd_state_event(
            "retry_decision",
            state,
            decision_reason=state.last_retry_decision_reason,
            base_token_index=base_token_index,
            fingerprint_attempts=fingerprint_attempts,
            unique_retry_fingerprints=unique_fingerprints,
        )
        short_circuit_retry, retry_quality_reason = (
            self._should_short_circuit_same_base_retry(
                state,
                base_token_index=base_token_index,
                fingerprint_attempts=fingerprint_attempts,
                unique_fingerprints=unique_fingerprints,
            )
        )
        if short_circuit_retry:
            state.low_value_retry_suppressed_count += 1
            state.same_base_retry_short_circuit_count += 1
            state.last_retry_quality_reason = retry_quality_reason
            state.last_retry_decision_reason = retry_quality_reason
            self._log_dasd_state_event(
                "retry_quality_suppressed",
                state,
                decision_reason=retry_quality_reason,
                base_token_index=base_token_index,
                fingerprint_attempts=fingerprint_attempts,
                unique_retry_fingerprints=unique_fingerprints,
            )
            return False, "low_value_retry_suppressed"

        send_ts = time.perf_counter()
        task = asyncio.create_task(
            self._validator.verify_bundle(
                client_id=self._dasd_client_id,
                request_id=state.request_id,
                bundle_id=bundle_id,
                epoch=state.epoch,
                base_token_index=base_token_index,
                token_ids=token_ids,
                timestamp_send_ms=int(time.time() * 1000),
                draft_model_id=config.draft_model,
            )
        )

        task_info = DasdTaskInfo(
            task=task,
            request_id=state.request_id,
            bundle_id=bundle_id,
            base_token_index=base_token_index,
            token_ids=token_ids,
            epoch_at_send=state.epoch,
            inflight_at_send=len(state.inflight) + 1,
            send_ts=send_ts,
            window_size_at_send=state.window_size,
            credit_at_send=(
                state.credit_controller.credit
                if state.credit_controller is not None
                else state.window_size
            ),
            tree_depth_at_send=(
                state.tree_budget.max_beam_len
                if state.tree_budget is not None
                else self._fixed_max_beam_len
            ),
            leaf_budget_at_send=(
                state.tree_budget.max_budget
                if state.tree_budget is not None
                else self._fixed_max_budget
            ),
            branch_width_at_send=(
                state.tree_budget.max_branch_width
                if state.tree_budget is not None
                else self._fixed_max_branch_width
            ),
        )
        state.inflight[bundle_id] = task_info
        state.task_to_bundle[task] = bundle_id
        state.next_bundle_id += 1
        state.next_base_index += state.window_size
        state.cleanup_induced_drain = False
        if state.cleanup_reason == "rollback_cleanup" and not state.frontier_sync_active:
            self._enter_frontier_sync(state, reason="rollback_cleanup_resume")
            state.frontier_sync_resume_count += 1
            self._log_dasd_state_event(
                "frontier_sync_resume",
                state,
                decision_reason="rollback_cleanup_resume",
                base_token_index=base_token_index,
                target_committed_len=state.frontier_sync_target_committed_len,
                remaining_progress=state.frontier_sync_progress_remaining,
            )
        if state.local_stabilization_active:
            self._log_dasd_state_event(
                "local_stabilization_retry",
                state,
                decision_reason=phase,
                base_token_index=base_token_index,
                fingerprint_attempts=fingerprint_attempts,
            )
        if config.dasd_debug:
            self._logger.info(
                "[DASD] send req=%s bundle=%d epoch=%d base=%d base_retry=%d W=%d inflight=%d credit=%s tree_depth=%s leaf_budget=%s branch_width=%s tokens=%s",
                state.request_id,
                bundle_id,
                state.epoch,
                base_token_index,
                state.base_retry_counts.get(base_token_index, 0),
                len(token_ids),
                len(state.inflight),
                task_info.credit_at_send,
                task_info.tree_depth_at_send,
                task_info.leaf_budget_at_send,
                task_info.branch_width_at_send,
                token_ids,
            )
        return True, "send_spawned"

    async def _receive_one_round_for_dasd(self, state: DasdRequestState):
        if state.aborted or not state.inflight:
            if config.dasd_debug and state.aborted:
                self._logger.info(
                    "[DASD] receive_skipped req=%s epoch=%d reason=request_inactive inflight=%d",
                    state.request_id,
                    state.epoch,
                    len(state.inflight),
                )
            return

        wait_tasks = list(state.task_to_bundle.keys())
        done, _ = await asyncio.wait(wait_tasks, return_when=asyncio.FIRST_COMPLETED)

        for task in done:
            bundle_id = state.task_to_bundle.pop(task, None)
            if bundle_id is None:
                continue

            task_info = state.inflight.pop(bundle_id, None)
            if task_info is None:
                state.late_task_drop_count += 1
                if config.dasd_debug:
                    self._logger.info(
                        "[DASD] late_task_dropped req=%s epoch=%d bundle=%s reason=cleanup_missing_inflight",
                        state.request_id,
                        state.epoch,
                        bundle_id,
                    )
                continue

            if task.cancelled():
                state.late_task_drop_count += 1
                if config.dasd_debug:
                    self._logger.info(
                        "[DASD] late_task_dropped req=%s epoch=%d bundle=%d base=%d reason=cancelled_after_cleanup",
                        state.request_id,
                        task_info.epoch_at_send,
                        task_info.bundle_id,
                        task_info.base_token_index,
                    )
                continue

            try:
                response = task.result()
                state.consecutive_rpc_failures = 0
                if (
                    state.frontier_sync_active
                    and task_info.base_token_index > state.committed_len
                ):
                    self._logger.warning(
                        "[DASD] frontier_sync_violation req=%s epoch=%d committed=%d task_base=%d bundle=%d",
                        state.request_id,
                        state.epoch,
                        state.committed_len,
                        task_info.base_token_index,
                        task_info.bundle_id,
                    )
                if (
                    task_info.request_id != state.request_id
                    or task_info.epoch_at_send != state.epoch
                ):
                    state.stale_epoch_drop_count += 1
                    if config.dasd_debug:
                        self._logger.info(
                            "[DASD] stale_response_ignored req=%s active_epoch=%d task_request=%s task_epoch=%d bundle=%d base=%d",
                            state.request_id,
                            state.epoch,
                            task_info.request_id,
                            task_info.epoch_at_send,
                            task_info.bundle_id,
                            task_info.base_token_index,
                        )
                    continue
            except grpc.aio.AioRpcError as e:
                status_code = e.code()
                if status_code == grpc.StatusCode.CANCELLED:
                    state.late_task_drop_count += 1
                    if config.dasd_debug:
                        self._logger.info(
                            "[DASD] late_task_dropped req=%s epoch=%d bundle=%d base=%d reason=grpc_cancelled",
                            state.request_id,
                            task_info.epoch_at_send,
                            task_info.bundle_id,
                            task_info.base_token_index,
                        )
                    continue
                state.consecutive_rpc_failures += 1
                state.rpc_failure_count += 1
                self._logger.warning(
                    "DASD bundle RPC failed (req=%s, bundle=%d, base=%d, status=%s)",
                    state.request_id,
                    bundle_id,
                    task_info.base_token_index,
                    status_code.name,
                )
                if status_code == grpc.StatusCode.UNAVAILABLE:
                    state.rpc_unavailable_abort_count += 1
                    await self._abort_dasd_request(
                        state,
                        reason="rpc_unavailable",
                    )
                    return
                if state.consecutive_rpc_failures >= config.dasd_abort_after_failures:
                    await self._abort_dasd_request(
                        state,
                        reason=(
                            "rpc_failures>="
                            f"{config.dasd_abort_after_failures}"
                        ),
                    )
                    return
                response = SimpleNamespace(
                    bundle_id=bundle_id,
                    accepted_len=0,
                    r_obs=0.0,
                    next_credit=state.window_size,
                    server_queue_delay_ms=0.0,
                    server_service_ms=0.0,
                    reject_reason=f"rpc_failure:{status_code.name}",
                    verifier_poisoned=False,
                    verifier_next_token_id=-1,
                    forced_commit_eligible=False,
                )
            except Exception:
                state.consecutive_rpc_failures += 1
                state.rpc_failure_count += 1
                self._logger.exception(
                    "DASD bundle task failed (req=%s, bundle=%d, base=%d). Treating as full rejection.",
                    state.request_id,
                    bundle_id,
                    task_info.base_token_index,
                )
                if state.consecutive_rpc_failures >= config.dasd_abort_after_failures:
                    await self._abort_dasd_request(
                        state,
                        reason=(
                            "rpc_failures>="
                            f"{config.dasd_abort_after_failures}"
                        ),
                    )
                    return
                response = SimpleNamespace(
                    bundle_id=bundle_id,
                    accepted_len=0,
                    r_obs=0.0,
                    next_credit=state.window_size,
                    server_queue_delay_ms=0.0,
                    server_service_ms=0.0,
                    reject_reason="rpc_failure",
                    verifier_poisoned=False,
                    verifier_next_token_id=-1,
                    forced_commit_eligible=False,
                )
            state.responses_by_base[task_info.base_token_index] = DasdBundleResult(
                response=response,
                recv_ts=time.perf_counter(),
                task_info=task_info,
            )
            if config.dasd_debug:
                self._logger.info(
                    "[DASD] recv req=%s bundle=%d base=%d accepted=%d",
                    state.request_id,
                    bundle_id,
                    task_info.base_token_index,
                    int(response.accepted_len),
                )

        self._commit_contiguous_for_dasd(state)

    def _commit_contiguous_for_dasd(self, state: DasdRequestState):
        while state.committed_len in state.responses_by_base:
            prev_committed = state.committed_len
            result = state.responses_by_base.pop(state.committed_len)
            task_info = result.task_info

            if task_info.epoch_at_send != state.epoch:
                continue

            verified_len = len(task_info.token_ids)
            accepted_len = max(0, min(int(result.response.accepted_len), verified_len))
            rtt_ms = (result.recv_ts - task_info.send_ts) * 1000.0
            next_credit = int(result.response.next_credit)
            reject_reason = str(getattr(result.response, "reject_reason", ""))
            verifier_poisoned = bool(
                getattr(result.response, "verifier_poisoned", False)
            )
            verifier_next_token_id = int(
                getattr(result.response, "verifier_next_token_id", 0)
            )
            forced_commit_eligible = bool(
                getattr(result.response, "forced_commit_eligible", False)
            )
            state.verify_rounds += 1
            state.sum_window_at_send += task_info.window_size_at_send
            state.sum_tree_depth_at_send += task_info.tree_depth_at_send
            state.sum_leaf_budget_at_send += task_info.leaf_budget_at_send
            control_feedback = None
            if state.credit_controller is not None:
                control_feedback = state.credit_controller.apply_feedback(
                    accepted_len=accepted_len,
                    proposed_len=verified_len,
                )
            self._refresh_dasd_control_targets(
                state,
                reason="verifier_response",
                next_credit=next_credit,
                feedback=control_feedback,
            )

            state.total_verified_tokens += verified_len
            state.total_accepted_tokens += accepted_len
            state.committed_len += accepted_len
            if accepted_len > 0:
                self._clear_suffix_refresh_anchor(state)
                self._record_frontier_sync_progress(
                    state,
                    reason="committed_progress",
                )
            assert state.committed_len >= prev_committed
            lifecycle = self._ensure_base_lifecycle(state, prev_committed)
            lifecycle["accepted_tokens_accumulated"] += accepted_len
            if accepted_len > 0:
                lifecycle["progressed"] = True
            if (
                state.rollback_blocked_committed_len is not None
                and state.committed_len > state.rollback_blocked_committed_len
            ):
                state.rollback_blocked_committed_len = None
                state.rollback_blocked_token_id = None
            if accepted_len == 0:
                state.consecutive_full_rejections += 1
                state.full_rejection_count += 1
                state.stall_rounds += 1
                lifecycle["full_rejections"] += 1
                state.per_base_max_full_rejection_count = max(
                    state.per_base_max_full_rejection_count,
                    lifecycle["full_rejections"],
                )
            else:
                state.consecutive_full_rejections = 0
            state.max_full_rejection_streak = max(
                state.max_full_rejection_streak,
                state.consecutive_full_rejections,
            )
            self._log_dasd_state_event(
                "rejection_progression",
                state,
                decision_reason=(
                    "full_rejection" if accepted_len == 0 else "partial_or_full_accept"
                ),
                base_token_index=prev_committed,
                verifier_next_token_id=verifier_next_token_id,
                reject_reason=reject_reason,
                accepted_len=accepted_len,
                proposed_len=verified_len,
            )
            forced_commit_applied = False
            if accepted_len > 0:
                self._exit_local_stabilization(
                    state,
                    reason="committed_progress",
                    success=True,
                )
                self._record_cooldown_progress(state, reason="committed_progress")
                self._exit_recovery_mode(state, reason="committed_progress")
                state.cleanup_induced_drain = False
            elif (
                config.dasd_recovery_mode_enabled
                and (
                    state.consecutive_full_rejections
                    >= config.dasd_recovery_full_rejection_threshold
                    or state.base_retry_counts.get(prev_committed, 0)
                    >= config.dasd_recovery_same_base_retry_threshold
                )
            ):
                trigger_reason = (
                    "full_rejection_streak"
                    if state.consecutive_full_rejections
                    >= config.dasd_recovery_full_rejection_threshold
                    else "same_base_retry"
                )
                self._enter_local_stabilization(state, reason=trigger_reason)
                state.rollback_cleanup_deferred_count += 1
                state.last_mitigation_decision_reason = trigger_reason
                self._log_dasd_state_event(
                    "rollback_cleanup_deferred",
                    state,
                    decision_reason=trigger_reason,
                    base_token_index=prev_committed,
                )
                self._enter_recovery_mode(state, reason=trigger_reason)
                self._record_rollback_cause(
                    state,
                    cause=(
                        "rollback_due_to_full_rejection_threshold"
                        if trigger_reason == "full_rejection_streak"
                        else "rollback_due_to_same_base_retry_threshold"
                    ),
                    base_token_index=prev_committed,
                    reject_reason=reject_reason,
                    verifier_next_token_id=verifier_next_token_id,
                )
            elif state.recovery_mode_active:
                state.recovery_mode_rounds_left = max(
                    0, state.recovery_mode_rounds_left - 1
                )
                if state.recovery_mode_rounds_left == 0:
                    self._exit_recovery_mode(state, reason="recovery_round_budget_exhausted")
            if self._should_force_commit(state, prev_committed, result.response):
                forced_commit_applied = self._apply_forced_commit(
                    state,
                    task_info=task_info,
                    response=result.response,
                )
            if config.dasd_debug:
                self._logger.info(
                    "[DASD] commit req=%s bundle=%d accepted=%d/%d committed=%d credit=%s W=%d tree_budget=%s",
                    state.request_id,
                    int(result.response.bundle_id),
                    accepted_len,
                    verified_len,
                    state.committed_len,
                    state.credit_controller.credit
                    if state.credit_controller is not None
                    else None,
                    state.window_size,
                    state.tree_budget,
                )

            self._result_logger.log(
                {
                    "type": "dasd_bundle",
                    "req_idx": state.req_idx,
                    "request_id": state.request_id,
                    "bundle_id": int(result.response.bundle_id),
                    "base_token_index": task_info.base_token_index,
                    "W": verified_len,
                    "send_ts": task_info.send_ts,
                    "recv_ts": result.recv_ts,
                    "rtt_ms": rtt_ms,
                    "epoch_at_send": task_info.epoch_at_send,
                    "inflight_at_send": task_info.inflight_at_send,
                    "credit_at_send": task_info.credit_at_send,
                    "tree_depth_at_send": task_info.tree_depth_at_send,
                    "leaf_budget_at_send": task_info.leaf_budget_at_send,
                    "branch_width_at_send": task_info.branch_width_at_send,
                    "accepted_len": accepted_len,
                    "r_obs": float(result.response.r_obs),
                    "next_credit": next_credit,
                    "verifier_next_token_id": verifier_next_token_id,
                    "forced_commit_eligible": forced_commit_eligible,
                    "forced_commit_applied": forced_commit_applied,
                    "credit_before": (
                        control_feedback.get("credit_before")
                        if control_feedback is not None
                        else None
                    ),
                    "credit_after": (
                        control_feedback.get("credit_after")
                        if control_feedback is not None
                        else None
                    ),
                    "strong_accept_streak": (
                        control_feedback.get("strong_accept_streak")
                        if control_feedback is not None
                        else None
                    ),
                    "full_rejection_streak": (
                        control_feedback.get("full_rejection_streak")
                        if control_feedback is not None
                        else None
                    ),
                    "chosen_window": state.window_size,
                    "chosen_tree_depth": (
                        state.tree_budget.max_beam_len
                        if state.tree_budget is not None
                        else None
                    ),
                    "chosen_leaf_budget": (
                        state.tree_budget.max_budget
                        if state.tree_budget is not None
                        else None
                    ),
                    "chosen_branch_width": (
                        state.tree_budget.max_branch_width
                        if state.tree_budget is not None
                        else None
                    ),
                    "server_queue_delay_ms": float(
                        result.response.server_queue_delay_ms
                    ),
                    "server_service_ms": float(result.response.server_service_ms),
                    "reject_reason": reject_reason,
                    "verifier_poisoned": verifier_poisoned,
                    "mode": "dasd",
                }
            )

            if forced_commit_applied:
                return

            if verifier_poisoned:
                self._dasd_server_poisoned = True
                state.aborted = True
                state.abort_reason = (
                    reject_reason if reject_reason else "verifier_poisoned"
                )
                abort_task = asyncio.create_task(
                    self._abort_dasd_request(state, reason=state.abort_reason)
                )
                abort_task.add_done_callback(lambda _: None)
                return

            if reject_reason.startswith("terminal_failed:"):
                state.aborted = True
                state.abort_reason = reject_reason
                abort_task = asyncio.create_task(
                    self._abort_dasd_request(state, reason=reject_reason)
                )
                abort_task.add_done_callback(lambda _: None)
                return

            if reject_reason.startswith("stale_epoch("):
                state.stale_epoch_drop_count += 1
            elif reject_reason.startswith("late_bundle_after_cleanup"):
                state.late_task_drop_count += 1

            if (
                accepted_len == 0
                and state.consecutive_full_rejections
                >= config.dasd_abort_after_failures
            ):
                abort_reason = (
                    "full_rejections>="
                    f"{config.dasd_abort_after_failures}"
                )
                state.aborted = True
                state.abort_reason = abort_reason
                abort_task = asyncio.create_task(
                    self._abort_dasd_request(state, reason=abort_reason)
                )
                abort_task.add_done_callback(lambda _: None)
                return

            if accepted_len < verified_len:
                state.rollbacks_count += 1
                failed_first_token = None
                if accepted_len < len(task_info.token_ids):
                    failed_first_token = int(task_info.token_ids[accepted_len])
                self._add_failure_cache_token(state, failed_first_token)
                rollback_cause = "rollback_due_to_post_recovery_rejection"
                if state.rollback_blocked_token_id is not None:
                    rollback_cause = "rollback_due_to_failure_cache_blocked_suffix"
                if state.recovery_mode_active:
                    rollback_cause = "rollback_due_to_post_recovery_rejection"
                self._record_rollback_cause(
                    state,
                    cause=rollback_cause,
                    base_token_index=prev_committed,
                    reject_reason=reject_reason,
                    verifier_next_token_id=verifier_next_token_id,
                    blocked_token=failed_first_token,
                    forced_commit_applied=forced_commit_applied,
                )
                state.rollback_blocked_committed_len = state.committed_len
                state.rollback_blocked_token_id = failed_first_token
                defer_rollback_cleanup = self._should_defer_rollback_cleanup(
                    state,
                    base_token_index=prev_committed,
                    rollback_cause=rollback_cause,
                )
                state.epoch += 1

                cancelled_task_infos = self._cancel_dasd_inflight(
                    state,
                    reason=(
                        "post_rejection_suffix_refresh"
                        if defer_rollback_cleanup
                        else "rollback_cleanup"
                    ),
                )
                if cancelled_task_infos:
                    drain_task = asyncio.create_task(
                        self._await_cancelled_dasd_tasks(
                            state,
                            cancelled_task_infos,
                            reason=(
                                "post_rejection_suffix_refresh"
                                if defer_rollback_cleanup
                                else "rollback_cleanup"
                            ),
                        )
                    )
                    drain_task.add_done_callback(lambda _: None)

                if defer_rollback_cleanup:
                    if not state.local_stabilization_active:
                        self._enter_local_stabilization(state, reason=rollback_cause)
                    state.last_mitigation_decision_reason = rollback_cause
                    self._log_dasd_state_event(
                        "mitigation_decision",
                        state,
                        decision_reason=rollback_cause,
                        base_token_index=prev_committed,
                    )
                    refreshed, refresh_reason = self._perform_suffix_refresh(
                        state,
                        reason=rollback_cause,
                    )
                    if refreshed:
                        self._log_dasd_state_event(
                            "recovery_resume_success",
                            state,
                            decision_reason=rollback_cause,
                            base_token_index=prev_committed,
                        )
                        return
                    state.rollback_cleanup_escalated_count += 1
                    self._log_dasd_state_event(
                        "rollback_cleanup_escalated",
                        state,
                        decision_reason=refresh_reason,
                        base_token_index=prev_committed,
                    )

                state.expensive_recovery_count += 1
                self._clear_suffix_refresh_anchor(state)
                self._prepare_dasd_refill_from_committed(
                    state, reason="rollback_cleanup"
                )
                self._exit_local_stabilization(
                    state,
                    reason="rollback_cleanup_escalated",
                    success=False,
                )
                if config.dasd_debug:
                    committed_prefix_tail = state.drafted_tokens[
                        max(0, state.committed_len - 8) : state.committed_len
                    ]
                    self._logger.info(
                        "[DASD] rollback req=%s epoch=%d committed=%d failed_first_token=%s credit=%s W=%d tree_budget=%s prefix_tail=%s prefix_tail_text=%r",
                        state.request_id,
                        state.epoch,
                        state.committed_len,
                        failed_first_token,
                        state.credit_controller.credit
                        if state.credit_controller is not None
                        else None,
                        state.window_size,
                        state.tree_budget,
                        committed_prefix_tail,
                        self._decode_dasd_debug_tokens(committed_prefix_tail),
                    )
                return

        if state.next_base_index < state.committed_len:
            state.next_base_index = state.committed_len

    async def _drain_dasd_inflight(self, state: DasdRequestState):
        if not state.inflight:
            return
        self._clear_suffix_refresh_anchor(state)
        task_infos = self._cancel_dasd_inflight(state, reason="request_finish_cleanup")
        await self._await_cancelled_dasd_tasks(
            state,
            task_infos,
            reason="request_finish_cleanup",
        )

    async def _abort_dasd_request(self, state: DasdRequestState, reason: str):
        if state.aborted and not state.inflight:
            return
        state.aborted = True
        self._clear_suffix_refresh_anchor(state)
        self._exit_frontier_sync(state, reason=reason, success=False)
        if state.abort_reason == "":
            state.abort_reason = reason
        self._logger.warning(
            "Aborting DASD request req=%s reason=%s committed=%d verified=%d accepted=%d",
            state.request_id,
            state.abort_reason,
            state.committed_len,
            state.total_verified_tokens,
            state.total_accepted_tokens,
        )
        state.finish_status = "explicit_abort_reason"
        await self._drain_dasd_inflight(state)

    def _draft_more_for_dasd(
        self,
        state: DasdRequestState,
        target_tokens: int,
        preserve_speculative_suffix_only: bool = False,
    ):
        if target_tokens <= 0:
            return

        max_attempt = 4
        appended = 0
        while appended < target_tokens and max_attempt > 0:
            max_attempt -= 1
            before_total_len = len(state.drafted_tokens)
            existing_suffix = state.drafted_tokens[state.committed_len :]
            if preserve_speculative_suffix_only:
                use_full_drafted = False
            else:
                use_full_drafted = len(state.drafted_tokens) > state.committed_len
            if config.dasd_debug:
                committed_prefix_tail = state.drafted_tokens[
                    max(0, state.committed_len - 8) : state.committed_len
                ]
                self._logger.info(
                    "[DASD] regrow_mode req=%s epoch=%d committed=%d drafted_len=%d target_tokens=%d credit=%s W=%d tree_budget=%s prefix_tail=%s drafted_suffix=%s preserve_speculative_suffix_only=%s use_full_drafted=%s",
                    state.request_id,
                    state.epoch,
                    state.committed_len,
                    len(state.drafted_tokens),
                    target_tokens,
                    state.credit_controller.credit
                    if state.credit_controller is not None
                    else None,
                    state.window_size,
                    state.tree_budget,
                    committed_prefix_tail,
                    existing_suffix,
                    preserve_speculative_suffix_only,
                    use_full_drafted,
                )
            self._reset_tree_from_dasd_state(
                state, use_full_drafted=use_full_drafted
            )
            with self._dasd_tree_budget_context(state) as tree_budget:
                self._grow_tree(prefill=True)
            self._log_dasd_leaf_candidates(state)
            path_tokens = self._extract_best_path_tokens(state)
            if config.dasd_debug:
                self._logger.info(
                    "[DASD] regrow_path req=%s epoch=%d committed=%d tree_budget=%s path_tokens=%s first_proposed=%s first_path_text=%r",
                    state.request_id,
                    state.epoch,
                    state.committed_len,
                    tree_budget,
                    path_tokens.tolist(),
                    path_tokens[0].item() if path_tokens.numel() > 0 else None,
                    self._decode_dasd_debug_tokens(path_tokens[:8].tolist()),
                )
            if path_tokens.numel() == 0:
                break

            candidate = path_tokens.tolist()
            candidate_len_before_trim = len(candidate)
            common = 0
            if preserve_speculative_suffix_only:
                max_common = min(len(existing_suffix), len(candidate))
                while common < max_common and existing_suffix[common] == candidate[common]:
                    common += 1
                candidate = candidate[common:]
                if config.dasd_debug:
                    self._logger.info(
                        "[DASD] regrow_overlap_trim req=%s epoch=%d committed=%d drafted_len=%d existing_suffix_len=%d candidate_len_before_trim=%d candidate_len_after_trim=%d common_prefix=%d novel_candidate=%s preserve_speculative_suffix_only=%s",
                        state.request_id,
                        state.epoch,
                        state.committed_len,
                        len(state.drafted_tokens),
                        len(existing_suffix),
                        candidate_len_before_trim,
                        len(candidate),
                        common,
                        candidate,
                        preserve_speculative_suffix_only,
                    )

            if not candidate:
                if config.dasd_debug:
                    self._logger.info(
                        "[DASD] regrow_no_novel_suffix req=%s epoch=%d committed=%d drafted_len=%d target_tokens=%d existing_suffix_len=%d candidate_len_before_trim=%d candidate_len_after_trim=%d preserve_speculative_suffix_only=%s reason=no_novel_suffix",
                        state.request_id,
                        state.epoch,
                        state.committed_len,
                        len(state.drafted_tokens),
                        target_tokens,
                        len(existing_suffix),
                        candidate_len_before_trim,
                        len(candidate),
                        preserve_speculative_suffix_only,
                    )
                break

            append_chunk = candidate[: max(0, target_tokens - appended)]
            if not append_chunk:
                break

            state.drafted_tokens.extend(append_chunk)
            appended += len(append_chunk)
            if config.dasd_debug:
                self._logger.info(
                    "[DASD] regrow_append req=%s epoch=%d committed=%d appended_tokens=%s new_drafted_len=%d preserve_speculative_suffix_only=%s",
                    state.request_id,
                    state.epoch,
                    state.committed_len,
                    append_chunk,
                    len(state.drafted_tokens),
                    preserve_speculative_suffix_only,
                )
            if preserve_speculative_suffix_only and len(state.drafted_tokens) == before_total_len:
                if config.dasd_debug:
                    self._logger.info(
                        "[DASD] regrow_no_novel_suffix req=%s epoch=%d committed=%d drafted_len=%d target_tokens=%d existing_suffix_len=%d candidate_len_before_trim=%d candidate_len_after_trim=%d preserve_speculative_suffix_only=%s reason=no_append_progress",
                        state.request_id,
                        state.epoch,
                        state.committed_len,
                        len(state.drafted_tokens),
                        target_tokens,
                        len(existing_suffix),
                        candidate_len_before_trim,
                        len(candidate),
                        preserve_speculative_suffix_only,
                    )
                break

    def _extract_best_path_tokens(self, state: Optional[DasdRequestState] = None):
        if self._tree.end <= self._tree.prefix_len:
            return torch.tensor([], dtype=torch.long, device=self._device)

        all_indices = torch.arange(
            self._tree.prefix_len, self._tree.end, dtype=torch.long, device=self._device
        )
        parent_mask = torch.zeros(self._tree.end, dtype=torch.bool, device=self._device)
        parent_mask[self._tree.parents[self._tree.prefix_len : self._tree.end]] = True
        leaf_indices = all_indices[~parent_mask[all_indices]]
        if leaf_indices.numel() == 0:
            leaf_indices = all_indices

        leaf_depths = self._tree.positions[leaf_indices]
        max_depth = leaf_depths.max()
        deepest = leaf_indices[leaf_depths == max_depth]
        if deepest.numel() == 1:
            best_leaf = deepest[0]
        else:
            best_leaf = deepest[self._tree.logprobs[deepest].argmax()]

        blocked_token = None
        blocked_tokens = set()
        filtered_leaf_indices = deepest
        filtered_first_tokens = None
        if state is not None:
            if (
                config.dasd_rollback_avoid_failed_token
                and state.rollback_blocked_committed_len == state.committed_len
                and state.rollback_blocked_token_id is not None
            ):
                blocked_token = int(state.rollback_blocked_token_id)
                blocked_tokens.add(blocked_token)

            blocked_tokens.update(self._blocked_tokens_for_prefix(state))

        if blocked_tokens:
            first_tokens = torch.tensor(
                [
                    self._first_continuation_token_for_leaf(int(leaf_idx.item()))
                    for leaf_idx in deepest
                ],
                dtype=torch.long,
                device=self._device,
            )
            blocked_tensor = torch.tensor(
                sorted(blocked_tokens), dtype=torch.long, device=self._device
            )
            keep_mask = ~torch.isin(first_tokens, blocked_tensor)
            filtered_first_tokens = first_tokens.tolist()
            if torch.any(keep_mask):
                filtered_leaf_indices = deepest[keep_mask]
                if filtered_leaf_indices.numel() == 1:
                    best_leaf = filtered_leaf_indices[0]
                else:
                    best_leaf = filtered_leaf_indices[
                        self._tree.logprobs[filtered_leaf_indices].argmax()
                    ]
                if config.dasd_debug:
                    self._logger.info(
                        "[DASD] rollback_filter req=%s epoch=%d committed=%d blocked_token=%s deepest_leaves=%s deepest_first_tokens=%s kept_leaves=%s",
                        state.request_id,
                        state.epoch,
                        state.committed_len,
                        sorted(blocked_tokens),
                        deepest.tolist(),
                        filtered_first_tokens,
                        filtered_leaf_indices.tolist(),
                    )
            elif config.dasd_debug:
                self._logger.info(
                    "[DASD] rollback_filter req=%s epoch=%d committed=%d blocked_token=%s no_alternative_leaf deepest_leaves=%s deepest_first_tokens=%s",
                    state.request_id,
                    state.epoch,
                    state.committed_len,
                    sorted(blocked_tokens),
                    deepest.tolist(),
                    filtered_first_tokens,
                )

        path_indices = []
        cursor = best_leaf
        while cursor >= self._tree.prefix_len:
            path_indices.append(cursor.item())
            cursor = self._tree.parents[cursor]

        if not path_indices:
            return torch.tensor([], dtype=torch.long, device=self._device)

        path_indices.reverse()
        path_tokens = self._tree.tokens[
            torch.tensor(path_indices, dtype=torch.long, device=self._device)
        ]
        if config.dasd_debug and state is not None:
            self._logger.info(
                "[DASD] best_path req=%s epoch=%d committed=%d best_leaf=%d blocked_token=%s path_indices=%s path_tokens=%s path_text=%r",
                state.request_id,
                state.epoch,
                state.committed_len,
                int(best_leaf.item()),
                sorted(blocked_tokens) if blocked_tokens else blocked_token,
                path_indices,
                path_tokens.tolist(),
                self._decode_dasd_debug_tokens(path_tokens[:8].tolist()),
            )
        return path_tokens

    def _reset_tree_from_dasd_state(
        self, state: DasdRequestState, use_full_drafted: bool = False
    ):
        prefix_generated_tokens = (
            state.drafted_tokens
            if use_full_drafted
            else state.drafted_tokens[: state.committed_len]
        )
        if prefix_generated_tokens:
            committed_tensor = torch.tensor(
                prefix_generated_tokens, dtype=torch.long, device=self._device
            ).unsqueeze(0)
            self._prefix_tokens = torch.cat(
                [self._initial_prefix_tokens, committed_tensor], dim=-1
            )
        else:
            self._prefix_tokens = self._initial_prefix_tokens.clone()

        if config.dasd_debug:
            prefix_tail = self._prefix_tokens[0, max(0, self._prefix_tokens.size(-1) - 8) :]
            self._logger.info(
                "[DASD] reset_tree req=%s epoch=%d committed=%d use_full_drafted=%s prefix_len=%d prefix_tail=%s",
                state.request_id,
                state.epoch,
                state.committed_len,
                use_full_drafted,
                self._prefix_tokens.size(-1),
                prefix_tail.tolist(),
            )

        self._engine.reset()
        self._tree = Tree(
            prefix_tokens=self._prefix_tokens,
            device=self._device,
            dtype=self._dtype,
            max_len=self._engine.max_len,
        )

    def _log_dasd_leaf_candidates(self, state: DasdRequestState):
        if not config.dasd_debug or self._tree.end <= self._tree.prefix_len:
            return

        all_indices = torch.arange(
            self._tree.prefix_len, self._tree.end, dtype=torch.long, device=self._device
        )
        parent_mask = torch.zeros(self._tree.end, dtype=torch.bool, device=self._device)
        parent_mask[self._tree.parents[self._tree.prefix_len : self._tree.end]] = True
        leaf_indices = all_indices[~parent_mask[all_indices]]
        if leaf_indices.numel() == 0:
            leaf_indices = all_indices

        leaf_depths = self._tree.positions[leaf_indices]
        leaf_scores = self._tree.logprobs[leaf_indices]
        max_depth = int(leaf_depths.max().item()) if leaf_depths.numel() > 0 else -1
        deepest = leaf_indices[leaf_depths == max_depth] if leaf_depths.numel() > 0 else leaf_indices
        if deepest.numel() == 0:
            best_leaf = None
        elif deepest.numel() == 1:
            best_leaf = deepest[0]
        else:
            best_leaf = deepest[self._tree.logprobs[deepest].argmax()]

        topk = min(8, leaf_indices.numel())
        order = torch.argsort(leaf_scores, descending=True)[:topk]
        summaries = []
        for idx in order.tolist():
            leaf_idx = int(leaf_indices[idx].item())
            first_cont_token = self._first_continuation_token_for_leaf(leaf_idx)
            summaries.append(
                {
                    "leaf_idx": leaf_idx,
                    "depth": int(leaf_depths[idx].item()),
                    "score": float(leaf_scores[idx].item()),
                    "first_token": first_cont_token,
                }
            )

        self._logger.info(
            "[DASD] leaf_candidates req=%s epoch=%d committed=%d num_leaves=%d best_leaf=%s top_leaves=%s",
            state.request_id,
            state.epoch,
            state.committed_len,
            int(leaf_indices.numel()),
            int(best_leaf.item()) if best_leaf is not None else None,
            summaries,
        )

    def _first_continuation_token_for_leaf(self, leaf_idx: int):
        cursor = torch.tensor(leaf_idx, dtype=torch.long, device=self._device)
        first_token = None
        while cursor >= self._tree.prefix_len:
            first_token = int(self._tree.tokens[cursor].item())
            parent = self._tree.parents[cursor]
            if parent < self._tree.prefix_len:
                break
            cursor = parent
        return first_token

    def _decode_dasd_debug_tokens(self, token_ids: list[int]):
        if not token_ids:
            return ""
        try:
            return self._tokenizer.decode(token_ids, skip_special_tokens=False)
        except Exception:
            return "<decode_error>"

    def _grow_tree(self, prefill: bool):
        self._logger.debug("Growing tree")

        # draft forward times
        draft_forward_times = []

        max_beam_len = self._max_beam_len
        if self._proactive_type == "included" and self._proactive_draft:
            max_beam_len = max(0, self._max_beam_len - config.proactive_max_beam_len)

        if torch.where(self._tree.status == self._tree.CANDIDATE)[0].numel() == 0:
            max_beam_len = 0

        for cnt in range(max_beam_len):
            self._logger.debug("Growing tree: %d / %d", cnt, max_beam_len)

            logits, beam_indices, beam_positions, beam_scores, draft_forward_t = (
                self._process_candidates(prefill)
            )
            prefill = False

            draft_forward_times.append(draft_forward_t)

            (
                next_beam_ids,
                next_beam_positions,
                next_beam_indices,
                beam_logprobs,
            ) = self._get_next_beams(
                logits=logits,
                beam_indices=beam_indices,
                beam_positions=beam_positions,
                beam_scores=beam_scores,
            )

            if next_beam_ids.numel() == 0:
                self._logger.debug("No more beams to grow")
                break

            if (
                self._tree.end - self._tree.prefix_len >= self._max_budget
                and not self._check_new_token_in_budget(beam_logprobs)
            ):
                self._logger.debug("Max budget reached. early stopping")
                break

            self._tree.add(
                token_ids=next_beam_ids,
                token_positions=next_beam_positions,
                parent_indices=next_beam_indices,
                logprobs=beam_logprobs,
            )

        if self._tree.end - self._tree.prefix_len >= self._max_budget:
            self._logger.debug("Trimming tree")
            self._trim_by_budget()

        return {"forward_t": draft_forward_times}

    def _process_candidates(self, warmup: bool):
        self._logger.debug("Processing candidates")
        candidate_indices = torch.where(
            self._tree.status[: self._tree.end] == self._tree.CANDIDATE
        )[0]

        if candidate_indices.numel() > self._max_n_beams:
            self._logger.debug("Choosing top %d candidates", self._max_n_beams)
            cumulative_logprobs = self._tree.logprobs[candidate_indices]
            top_k_indices = cumulative_logprobs.topk(
                k=self._max_n_beams, sorted=False
            ).indices
            candidate_indices = candidate_indices[top_k_indices]
            candidate_indices, _ = candidate_indices.sort()

        if warmup:
            prefill_input_indices = torch.arange(
                candidate_indices.min().item(), device=self._device
            )
            prefill_input_ids = self._tree.tokens[prefill_input_indices].unsqueeze(0)
            prefill_position_ids = self._tree.positions[
                prefill_input_indices
            ].unsqueeze(0)
            prefill_cache_seq_indices = prefill_input_indices
            prefill_attention_mask = self._tree.amask[..., prefill_input_indices, :]

            self._engine.prefill(
                input_ids=prefill_input_ids,
                position_ids=prefill_position_ids,
                batch_idx=0,
                cache_seq_indices=prefill_cache_seq_indices,
                attention_mask=prefill_attention_mask,
            )

        input_indices = candidate_indices

        input_ids = self._tree.tokens[input_indices].unsqueeze(0)
        position_ids = self._tree.positions[input_indices].unsqueeze(0)
        cache_seq_indices = input_indices
        cache_batch_indices = torch.full_like(
            cache_seq_indices, 0, dtype=torch.long, device=self._device
        )
        attention_mask = self._tree.amask[..., input_indices, :]

        with util.Timing(device=self._device, mode=self._draft_forward_time_mode) as t:
            logits = self._engine.forward(
                input_ids=input_ids,
                position_ids=position_ids,
                cache_batch_indices=cache_batch_indices,
                cache_seq_indices=cache_seq_indices,
                attention_mask=attention_mask,
            )

        self._tree.status[candidate_indices] = self._tree.PROCESSED
        beam_scores = self._tree.logprobs[candidate_indices]
        beam_positions = self._tree.positions[candidate_indices]
        logits = logits[0, -candidate_indices.size(-1) :, :]

        return (logits, candidate_indices, beam_positions, beam_scores, t.elapsed)

    def _get_next_beams(
        self,
        logits: torch.Tensor,
        beam_indices: torch.Tensor,
        beam_positions: torch.Tensor,
        beam_scores: torch.Tensor,
    ):
        self._logger.debug("Getting next beams")
        DECAY_FACTOR = np.log(0.9)

        logprobs = torch.log_softmax(logits, dim=-1)  # shape: [n_beams, vocab_size]
        logprobs_k = logprobs.topk(
            k=self._max_branch_width, dim=-1, sorted=False
        )  # shape: [n_beams, max_branch_width]
        leaves_ids = logprobs_k.indices
        leaves_probs = logprobs_k.values

        flat_incoming_probs = (
            beam_scores.unsqueeze(-1) + DECAY_FACTOR + leaves_probs
        ).flatten()
        flat_incoming_ids = leaves_ids.flatten()

        joint_probs = torch.concat(
            [
                self._tree.logprobs[self._tree.prefix_len : self._tree.end],
                flat_incoming_probs,
            ]
        )

        if (
            joint_probs.size(-1) > self._max_budget
            or joint_probs.size(-1) + (self._tree.end - self._tree.prefix_len)
            > self._max_len
        ):
            min_joint_prob = joint_probs.topk(
                k=self._max_budget, sorted=False, dim=-1
            ).values.min()

            flat_best_mask = torch.where(flat_incoming_probs >= min_joint_prob)[0]
            flat_best_probs = flat_incoming_probs[flat_best_mask]
            flat_best_indices = flat_best_mask
            best_children_token_ids = flat_incoming_ids[flat_best_indices]

            if flat_best_indices.size(-1) + self._tree.end > self._max_len:
                raise NotImplementedError("Implement trim budget")

        else:
            flat_best_probs = flat_incoming_probs
            flat_best_indices = torch.arange(
                flat_incoming_probs.size(0), device=logits.device
            )
            best_children_token_ids = flat_incoming_ids

        best_hypo_ids = flat_best_indices // self._max_branch_width
        best_beam_indices = beam_indices[best_hypo_ids]
        best_children_positions = beam_positions[best_hypo_ids] + 1

        return (
            best_children_token_ids,
            best_children_positions,
            best_beam_indices,
            flat_best_probs,
        )

    def _check_new_token_in_budget(self, cumulative_beam_scores: torch.Tensor):
        lowest_tree_logprob = (
            self._tree.logprobs[self._tree.prefix_len : self._tree.end]
            .topk(k=self._max_budget, dim=-1, sorted=False)
            .values.min()
        )
        best_new_logprob = cumulative_beam_scores.max()

        return best_new_logprob >= lowest_tree_logprob

    def _trim_by_budget(self):
        src_indices = (
            self._tree.logprobs[self._tree.prefix_len : self._tree.end]
            .topk(k=self._max_budget, sorted=False)
            .indices
            + self._tree.prefix_len
        )
        dest_indices = torch.arange(
            self._tree.prefix_len,
            self._tree.prefix_len + src_indices.size(-1),
            device=self._device,
        )

        self._tree.gather(src_indices, dest_indices)
        self._engine.gather(src_indices, dest_indices)

    async def _validate_tree(self, req_idx: int, prefill=False):
        self._logger.debug("Validating tree")

        with util.Timing(
            device=self._device, mode=self._target_time_mode
        ) as preprocess_t:
            target_token_map_bool = (
                self._tree.status[: self._tree.end] >= self._tree.PROCESSED
            )
            target_token_map_bool[: self._tree.prefix_len] = False
            target_token_indices = torch.where(target_token_map_bool)[0]
            target_parent_indices = self._tree.parents[: self._tree.end][
                target_token_map_bool
            ]

            input_token_map_bool = target_token_map_bool.clone()
            input_token_map_bool[target_parent_indices] = True

            input_ids = self._tree.tokens[: self._tree.end][
                input_token_map_bool
            ].unsqueeze(0)
            position_ids = self._tree.positions[: self._tree.end][
                input_token_map_bool
            ].unsqueeze(0)
            cache_seq_indices = torch.where(input_token_map_bool)[0]
            attention_mask = self._tree.amask[..., cache_seq_indices, :]

        with util.Timing(device=self._device, mode=self._target_time_mode) as wait_t:
            prefix = self._prompt if prefill else None
            target_result = asyncio.create_task(
                self._validator.request(
                    client_idx=self._client_idx,
                    req_idx=req_idx,
                    input_ids=input_ids,
                    position_ids=position_ids,
                    cache_seq_indices=cache_seq_indices,
                    attention_mask=attention_mask,
                    parent_indices=target_parent_indices,
                    prefill=prefill,
                    prefix=prefix,
                )
            )
            await asyncio.sleep(0.00001)

            if self._proactive_client is not None:
                (
                    root_leaf_idx,
                    root_token_id,
                    proactive_tree_prefix_len,
                    proactive_tree_end,
                ) = self._proactive_client.draft()

            selection, prefill_cnt = (
                target_result.result() if target_result.done() else await target_result
            )

        with util.Timing(
            device=self._device, mode=self._target_time_mode
        ) as postprocess_t:
            interim_t = torch.ones_like(self._tree.tokens[: self._tree.end])
            interim_t[input_token_map_bool] = selection

            draft_token_choices = self._tree.tokens[: self._tree.end][
                target_token_map_bool
            ]
            target_token_choices = interim_t[target_parent_indices]

            accept_flags = draft_token_choices == target_token_choices

            accept_indices = target_token_indices[accept_flags]

            accept_mask = torch.zeros(self._tree.end, device=self._device)
            accept_mask[: self._tree.prefix_len] = 1
            accept_mask[accept_indices] = 1
            accepted_amask = attention_mask[0, 0, :, : self._tree.end] * accept_mask

            mask_row_sums = (
                attention_mask[0, 0, :, : self._tree.end].sum(dim=1).to(torch.long)
            )

            seq_lengths = accepted_amask.sum(dim=1).to(torch.long)
            best_seq_idx = (mask_row_sums * (mask_row_sums == seq_lengths)).argmax()
            best_seq_mask = attention_mask[0, 0, best_seq_idx, : self._tree.end].to(
                torch.bool
            )

            fresh_token_indices = (
                torch.where(best_seq_mask[self._tree.prefix_len :])[0]
                + self._tree.prefix_len
            )
            fresh_token_ids = self._tree.tokens[fresh_token_indices]

            last_accepted_token_idx = (
                fresh_token_indices[-1]
                if fresh_token_indices.numel() > 0
                else torch.tensor([self._tree.prefix_len - 1])
            ).to(self._device)

            # add one bonus token to num of accepted tokens
            self._logger.debug(
                "Num of accepted tokens: %d", fresh_token_indices.numel() + 1
            )

            extra_token_id = torch.tensor(
                [interim_t[last_accepted_token_idx]], device=self._device
            )

            if self._proactive_client is not None:
                self._previous_proactive_draft = self._proactive_draft

            if (
                self._proactive_client is not None
                and root_leaf_idx is not None  # type: ignore
                and root_leaf_idx == last_accepted_token_idx  # type: ignore
                and extra_token_id == root_token_id  # type: ignore
            ):
                self._proactive_draft = True
                self._reorder_by_sequence_proactive(
                    best_seq_mask,
                    proactive_tree_prefix_len,  # type: ignore
                    proactive_tree_end,  # type: ignore
                )
            else:
                self._proactive_draft = False
                self._reorder_by_sequence(best_seq_mask)
                self._tree.add(
                    token_ids=extra_token_id,
                    token_positions=self._tree.positions[self._tree.end - 1] + 1,
                    parent_indices=torch.tensor(
                        [self._tree.end - 1], device=self._device
                    ),
                    logprobs=torch.tensor([0.0], device=self._device),
                )
                self._tree.prefix_len = self._tree.end
                self._tree.status[: self._tree.prefix_len - 1] = self._tree.PROMPT

            fresh_token_ids = torch.cat(
                [fresh_token_ids, extra_token_id], dim=-1
            ).unsqueeze(0)

        stats = {
            "preprocess_t": preprocess_t.elapsed,
            "wait_t": wait_t.elapsed,
            "postprocess_t": postprocess_t.elapsed,
            "num_accepted_tokens": fresh_token_ids.size(-1),
            "prefill": prefill_cnt,
            "previous_proactive": self._previous_proactive_draft
            if self._proactive_client
            else False,
            "proactive": self._proactive_draft if self._proactive_client else False,
        }

        return fresh_token_ids, stats

    def _reorder_by_sequence(self, seq_mask: torch.Tensor):
        """
        Reorder the tree and engine's kv cache according to the validated sequence.

        Args:
            seq_mask: Sequence Mask
        """

        seq_indices = torch.where(seq_mask != 0)[0]

        self._engine.gather(
            seq_indices,
            torch.arange(seq_indices.size(-1), device=self._device),
        )

        self._tree.reorder_by_sequence(seq_mask, seq_indices)

    def _reorder_by_sequence_proactive(
        self,
        seq_mask: torch.Tensor,
        proactive_tree_prefix_len: int,
        proactive_tree_end: int,
    ):
        """
        Reorders the tree and engine's kv cache in a valid sequence
        when the tree generated by Proactive Draft is valid.

        Args:
            seq_mask: Sequence Mask
            proactive_tree_prefix_len: Start of the tree generated by Proactive Draft
            proactive_tree_end: End of the tree generated by Proactive Draft
        """
        seq_indices = torch.where(seq_mask != 0)[0]
        max_src_idx = proactive_tree_end
        mapping_tensor = torch.full(
            (max_src_idx,), -1, dtype=torch.long, device=self._device
        )

        # Original Draft Tree

        new_prefix_len = int(torch.sum(seq_mask).item())
        if torch.any(seq_mask[self._tree.prefix_len :]):
            src_indices = seq_indices[seq_indices >= self._tree.prefix_len]
            dest_indices = torch.arange(
                self._tree.prefix_len, new_prefix_len, device=self._device
            )
            mapping_tensor[src_indices] = dest_indices

            self._tree.tokens[dest_indices] = self._tree.tokens[src_indices]
            self._tree.positions[dest_indices] = dest_indices
            self._tree.parents[dest_indices] = dest_indices - 1
            self._tree.status[dest_indices] = self._tree.GENERATED

        # Proactive Tree

        src_indices = torch.arange(
            proactive_tree_prefix_len, proactive_tree_end, device=self._device
        )
        dest_indices = torch.arange(
            new_prefix_len,
            new_prefix_len + proactive_tree_end - proactive_tree_prefix_len,
            device=self._device,
        )
        mapping_tensor[src_indices] = dest_indices

        self._tree.tokens[dest_indices] = self._tree.tokens[src_indices]
        self._tree.positions[dest_indices] = self._tree.positions[src_indices]
        self._tree.parents[dest_indices] = mapping_tensor[
            self._tree.parents[src_indices]
        ]
        self._tree.status[dest_indices] = self._tree.status[src_indices]
        self._tree.logprobs[dest_indices] = self._tree.logprobs[src_indices]
        self._tree.amask[
            ...,
            dest_indices,
            new_prefix_len : new_prefix_len
            + proactive_tree_end
            - proactive_tree_prefix_len,
        ] = self._tree.amask[
            ..., src_indices, proactive_tree_prefix_len:proactive_tree_end
        ]

        self._tree.end = new_prefix_len + proactive_tree_end - proactive_tree_prefix_len
        self._tree.prefix_len = new_prefix_len + 1

        self._tree.status[: self._tree.prefix_len - 1] = self._tree.PROMPT
        self._tree.status[self._tree.prefix_len - 1 : self._tree.prefix_len + 1] = (
            self._tree.PROCESSED
        )
        self._tree.status[self._tree.status == self._tree.POST_CANDIDATE] = (
            self._tree.CANDIDATE
        )
        self._tree.status[self._tree.status == self._tree.POST_PROCESSED] = (
            self._tree.PROCESSED
        )

        self._tree.logprobs[self._tree.end :].zero_()
        # FIXME: change to public property access
        self._tree._data[:, self._tree.end :].zero_()

        _causal_mask = torch.tril(
            torch.ones(
                self._tree.prefix_len,
                self._tree.prefix_len,
                dtype=self._dtype,
                device=self._device,
            )
        )
        self._tree.amask[..., : self._tree.prefix_len, : self._tree.prefix_len] = (
            _causal_mask
        )
        self._tree.amask[
            ..., self._tree.prefix_len : self._tree.end, : self._tree.prefix_len
        ] = 1.0

        src_indices = seq_mask[: self._tree.prefix_len]
        src_indices = torch.where(src_indices)[0]
        dst_indices = torch.arange(src_indices.size(-1), device=self._device)

        self._engine.gather(src_indices, dst_indices)
    @property
    def dasd_server_poisoned(self):
        return self._dasd_server_poisoned
