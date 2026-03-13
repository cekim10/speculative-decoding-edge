import asyncio
import math
import multiprocessing as mp
import os
import queue
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import grpc
import torch
from rich.progress import track

import log
import util
from config import SpecEdgeBatchServerConfig as config
from specedge.engine.graph import BatchGraphEngine
from specedge_grpc import specedge_pb2, specedge_pb2_grpc


@dataclass
class DasdClientControlState:
    ema_acceptance: float
    last_credit: int
    inflight_count: int = 0
    total_verified_tokens: int = 0
    total_accepted_tokens: int = 0


@dataclass
class DasdVerifierState:
    client_id: str
    slot_idx: int
    request_id: str
    epoch: int
    prompt_len: int
    committed_tokens: list[int]
    next_token_id: int
    cleanup_safe: bool = True


class SpecExecBatchServer(specedge_pb2_grpc.SpecEdgeServiceServicer):
    def __init__(
        self,
        shutdown_event: asyncio.Event = None,
    ) -> None:
        self._logger = log.get_logger()
        self._result_logger = log.get_result_logger()

        self._loop = asyncio.get_event_loop()
        self._synced = 0
        self._num_clients = config.num_clients
        self._all_sync = asyncio.Condition()
        self._mode = config.mode
        self._dasd_mode_enabled = self._mode == "dasd" and config.dasd_enable_async
        self._mp_ctx = mp.get_context("spawn")

        self._shutdown_event = shutdown_event
        self._resp_queue_task = None
        self._inference_process = None
        self._resp_lock = threading.Lock()

        self._recv_queue = None
        self._resp_queue = None
        self._resp_futures: dict[int, asyncio.Future] = {}

        self._dasd_recv_queue = None
        self._dasd_resp_queue = None
        self._dasd_futures: dict[tuple[str, str, int], asyncio.Future] = {}
        self._dasd_client_state: dict[str, DasdClientControlState] = {}
        self._dasd_verifier_poisoned = False

        if self._dasd_mode_enabled:
            self._logger.info("Initializing DASD verifier mode")
            self._dasd_recv_queue = self._mp_ctx.Queue()
            self._dasd_resp_queue = self._mp_ctx.Queue()
            self._resp_queue_task = self._loop.create_task(
                self._init_dasd_resp_queue_loop()
            )
            self._init_dasd_inference_loop()
        else:
            self._logger.info("Initializing baseline SpecEdge server mode")
            self._recv_queue = self._mp_ctx.Queue()
            self._resp_queue = self._mp_ctx.Queue()
            self._resp_queue_task = self._loop.create_task(self._init_resp_queue_loop())
            self._init_inference_loop()

    async def _init_resp_queue_loop(self):
        self._logger.debug("Starting response queue loop")
        while True:
            try:
                if self._shutdown_event and self._shutdown_event.is_set():
                    self._logger.info("Response queue loop shutting down...")
                    break

                try:
                    if self._resp_queue is None:
                        break
                    raw_data, client_idx = await self._loop.run_in_executor(
                        None, self._resp_queue.get, True, 0.5
                    )
                except queue.Empty:
                    continue

                if raw_data is None and client_idx == -1:
                    self._logger.info(
                        "Received shutdown sentinel, stopping response queue loop"
                    )
                    break

                self._logger.debug("Received response for client %d", client_idx)

                with self._resp_lock:
                    if client_idx in self._resp_futures:
                        future = self._resp_futures.pop(client_idx)
                        future.set_result(raw_data)
                    else:
                        self._logger.error("Client index not found in futures")
            except Exception as e:
                self._logger.error("Error processing response: %s", e)
                if self._shutdown_event and self._shutdown_event.is_set():
                    break

    async def _init_dasd_resp_queue_loop(self):
        self._logger.debug("Starting DASD response queue loop")
        while True:
            try:
                if self._shutdown_event and self._shutdown_event.is_set():
                    self._logger.info("DASD response queue loop shutting down...")
                    break

                try:
                    if self._dasd_resp_queue is None:
                        break
                    key, payload = await self._loop.run_in_executor(
                        None, self._dasd_resp_queue.get, True, 0.5
                    )
                except queue.Empty:
                    continue

                if key is None and payload is None:
                    self._logger.info(
                        "Received DASD shutdown sentinel; stopping response loop"
                    )
                    break

                with self._resp_lock:
                    if key in self._dasd_futures:
                        future = self._dasd_futures.pop(key)
                        if not future.done():
                            future.set_result(payload)
                    else:
                        self._logger.warning("Stale DASD response received for key=%s", key)
            except Exception as e:
                self._logger.error("Error processing DASD response: %s", e)
                if self._shutdown_event and self._shutdown_event.is_set():
                    break

    async def Sync(self, request, context):
        async with self._all_sync:
            self._synced += 1

            if self._synced == self._num_clients:
                self._synced = 0
                self._all_sync.notify_all()
            else:
                await self._all_sync.wait()

        return specedge_pb2.SyncResponse()

    async def Validate(self, request, context):
        if self._dasd_mode_enabled:
            await context.abort(
                grpc.StatusCode.FAILED_PRECONDITION,
                "Validate RPC is disabled in DASD mode. Use VerifyBundleAsync.",
            )

        if self._recv_queue is None:
            await context.abort(
                grpc.StatusCode.INTERNAL, "Baseline receive queue is not initialized."
            )

        self._logger.info("Received request: %s", request.client_idx)
        fut = asyncio.Future()

        with self._resp_lock:
            self._resp_futures[request.client_idx] = fut

        self._recv_queue.put(request.SerializeToString())
        selection, prefill_cnt = await asyncio.wait_for(fut, timeout=5.0)
        return specedge_pb2.ValidateResponse(selection=selection, prefill=prefill_cnt)

    async def VerifyBundleAsync(self, request, context):
        if not self._dasd_mode_enabled:
            await context.abort(
                grpc.StatusCode.FAILED_PRECONDITION,
                "VerifyBundleAsync RPC is available only when mode=dasd and enable_async=true.",
            )

        if self._dasd_recv_queue is None:
            await context.abort(
                grpc.StatusCode.INTERNAL, "DASD receive queue is not initialized."
            )

        client_id = request.client_id.strip()
        if client_id == "":
            await context.abort(
                grpc.StatusCode.INVALID_ARGUMENT, "client_id must be non-empty."
            )

        key = (client_id, request.request_id, int(request.bundle_id))
        state = self._dasd_client_state.setdefault(
            client_id,
            DasdClientControlState(
                ema_acceptance=1.0,
                last_credit=config.dasd_w_min,
            ),
        )
        state.inflight_count += 1

        send_ts = time.perf_counter()
        fut = asyncio.Future()
        with self._resp_lock:
            self._dasd_futures[key] = fut

        self._dasd_recv_queue.put((send_ts, request.SerializeToString()))

        try:
            payload: dict[str, Any] = await asyncio.wait_for(fut, timeout=60.0)
        except asyncio.TimeoutError:
            with self._resp_lock:
                self._dasd_futures.pop(key, None)
            state.inflight_count = max(0, state.inflight_count - 1)
            await context.abort(grpc.StatusCode.DEADLINE_EXCEEDED, "Bundle verify timed out.")

        state.inflight_count = max(0, state.inflight_count - 1)

        accept_bitmap = list(payload["accept_bitmap"])
        verified_tokens = len(accept_bitmap)
        accepted_len = int(payload["accepted_len"])
        r_obs = (
            float(payload.get("r_obs", 0.0))
            if verified_tokens == 0
            else accepted_len / verified_tokens
        )
        loss_event = accepted_len < verified_tokens

        if verified_tokens > 0:
            decay = config.dasd_ema_decay
            state.ema_acceptance = (
                decay * state.ema_acceptance + (1.0 - decay) * r_obs
            )
        next_credit = self._apply_aimd_credit(state.last_credit, loss_event)
        state.last_credit = next_credit
        state.total_verified_tokens += verified_tokens
        state.total_accepted_tokens += accepted_len

        end_ts = time.perf_counter()
        server_rtt_ms = (end_ts - send_ts) * 1000.0
        queue_delay_ms = float(payload.get("queue_delay_ms", 0.0))
        service_ms = float(payload.get("server_service_ms", 0.0))
        queue_depth = int(payload.get("queue_depth", -1))
        reject_reason = str(payload.get("reject_reason", ""))
        verifier_poisoned = bool(payload.get("verifier_poisoned", False))
        verifier_next_token_id = int(payload.get("verifier_next_token_id", 0))
        forced_commit_eligible = bool(payload.get("forced_commit_eligible", False))

        self._result_logger.log(
            {
                "mode": "dasd",
                "dasd": {
                    "client_id": client_id,
                    "request_id": request.request_id,
                    "bundle_id": int(request.bundle_id),
                    "base_token_index": int(request.base_token_index),
                    "window_size": verified_tokens,
                    "accepted_len": accepted_len,
                    "goodput": r_obs,
                    "loss_event": loss_event,
                    "next_credit": next_credit,
                    "queue_delay_ms": queue_delay_ms,
                    "service_ms": service_ms,
                    "server_rtt_ms": server_rtt_ms,
                    "queue_depth": queue_depth,
                    "inflight_count": state.inflight_count,
                    "ema_acceptance": state.ema_acceptance,
                    "reject_reason": reject_reason,
                    "verifier_poisoned": verifier_poisoned,
                    "total_verified_tokens": state.total_verified_tokens,
                    "total_accepted_tokens": state.total_accepted_tokens,
                },
            }
        )

        return specedge_pb2.VerifyBundleResponse(
            request_id=request.request_id,
            bundle_id=request.bundle_id,
            base_token_index=request.base_token_index,
            accept_bitmap=accept_bitmap,
            accepted_len=accepted_len,
            r_obs=r_obs,
            next_credit=next_credit,
            server_queue_delay_ms=queue_delay_ms,
            server_service_ms=service_ms,
            reject_reason=reject_reason,
            verifier_poisoned=verifier_poisoned,
            verifier_next_token_id=verifier_next_token_id,
            forced_commit_eligible=forced_commit_eligible,
        )

    def _apply_aimd_credit(self, current_credit: int, loss_event: bool):
        if loss_event:
            return max(config.dasd_w_min, int(math.floor(config.dasd_beta * current_credit)))
        return min(config.dasd_w_max, current_credit + config.dasd_alpha)

    def _init_inference_loop(self):
        if self._recv_queue is None or self._resp_queue is None:
            raise RuntimeError("Baseline queues are not initialized")
        self._inference_process = self._mp_ctx.Process(
            target=_init_inference,
            args=(
                self._num_clients,
                self._recv_queue,
                self._resp_queue,
            ),
            daemon=False,
        )
        self._inference_process.start()

    def _init_dasd_inference_loop(self):
        if self._dasd_recv_queue is None or self._dasd_resp_queue is None:
            raise RuntimeError("DASD queues are not initialized")
        self._inference_process = self._mp_ctx.Process(
            target=_init_dasd_inference,
            args=(
                self._num_clients,
                self._dasd_recv_queue,
                self._dasd_resp_queue,
            ),
            daemon=False,
        )
        self._inference_process.start()

    async def cleanup(self):
        """Clean up resources during shutdown"""
        self._logger.info("Starting cleanup...")
        is_dasd = self._dasd_mode_enabled

        try:
            self._logger.info("Sending shutdown signal to inference process...")
            if is_dasd:
                if self._dasd_recv_queue is not None:
                    self._dasd_recv_queue.put(None)
            elif self._recv_queue is not None:
                self._recv_queue.put(None)
        except Exception as e:
            self._logger.exception("Error sending shutdown signal %s", e)

        # Wait for inference process to finish (with timeout)
        if self._inference_process and self._inference_process.is_alive():
            self._logger.info("Waiting for inference process to terminate...")
            self._inference_process.join(timeout=10.0)

            if self._inference_process.is_alive():
                self._logger.warning("Inference process did not terminate, forcing...")
                self._inference_process.terminate()
                self._inference_process.join(timeout=2.0)

                if self._inference_process.is_alive():
                    self._logger.error("Inference process still alive, killing...")
                    self._inference_process.kill()

        # Send sentinel to response queue to stop the loop
        try:
            if is_dasd:
                if self._dasd_resp_queue is not None:
                    self._dasd_resp_queue.put((None, None))
            elif self._resp_queue is not None:
                self._resp_queue.put((None, -1))
        except Exception as e:
            self._logger.error(f"Error sending sentinel to response queue: {e}")

        # Wait for response queue task to complete
        if self._resp_queue_task and not self._resp_queue_task.done():
            self._logger.info("Waiting for response queue task to complete...")
            try:
                await asyncio.wait_for(self._resp_queue_task, timeout=2.0)
            except asyncio.TimeoutError:
                self._logger.warning("Response queue task did not complete in time")
                self._resp_queue_task.cancel()

        # Close queues
        try:
            if self._recv_queue is not None:
                self._recv_queue.close()
                self._recv_queue.join_thread()
            if self._resp_queue is not None:
                self._resp_queue.close()
                self._resp_queue.join_thread()
            if self._dasd_recv_queue is not None:
                self._dasd_recv_queue.close()
                self._dasd_recv_queue.join_thread()
            if self._dasd_resp_queue is not None:
                self._dasd_resp_queue.close()
                self._dasd_resp_queue.join_thread()
        except Exception as e:
            self._logger.exception("Error closing queue %s", e)

        self._logger.info("Cleanup complete")


class InferenceController:
    def __init__(
        self,
        num_clients: int,
        recv_queue: mp.Queue,
        resp_queue: mp.Queue,
    ) -> None:
        self._logger = log.get_logger()
        self._result_logger = log.get_result_logger()

        self._dtype = config.dtype
        self._device = config.device

        self._num_clients = num_clients
        self._temperature = config.temperature
        self._batch_size = config.max_batch_size
        self._max_budget = config.max_budget
        self._max_n_beams = self._max_budget + 1
        self._max_len = config.max_len
        self._batch_type = config.batch_type
        self.dataset = util.load_dataset(config.dataset, config.target_model)

        self._request_batches: list[specedge_pb2.ValidateRequest] = []
        self._recv_queue = recv_queue
        self._resp_queue = resp_queue

        self._tokenizer = util.load_tokenizer(config.target_model)

        self._logger.info("Initializing inference controller")

        self._logger.debug("Loading model")
        self._model = util.load_graph_model(
            name=config.target_model,
            device=config.device,
            dtype=config.dtype,
        )

        self._engine = BatchGraphEngine(
            model=self._model,
            max_len=config.max_len,
            max_batch_size=config.max_batch_size,
            max_n_beams=self._max_n_beams,
        )

        self.k_cache = torch.zeros(
            (
                self._model.config.num_hidden_layers,
                self._num_clients,
                self._model.config.num_key_value_heads,
                self._max_len,
                self._model.config.head_dim,
            ),
            dtype=self._dtype,
            device=self._device,
        )

        self.v_cache = torch.zeros_like(
            self.k_cache, dtype=self._dtype, device=self._device
        )

        self._client_indices = torch.zeros(
            (self._batch_size,),
            dtype=torch.long,
            device=self._device,
        )

        self._iter_idx = torch.zeros(
            (self._num_clients,),
            dtype=torch.long,
            device=self._device,
        )

        self._input_ids = torch.zeros(
            (self._batch_size, self._max_n_beams),
            dtype=torch.long,
            device=self._device,
        )

        self._parent_indices = torch.zeros(
            (self._batch_size, self._max_budget), dtype=torch.long, device=self._device
        )

        self._position_ids = torch.zeros(
            (self._batch_size, self._max_n_beams),
            dtype=torch.long,
            device=self._device,
        )

        self._cache_batch_indices = torch.arange(
            self._batch_size, dtype=torch.long, device=self._device
        ).repeat_interleave(self._max_n_beams)

        self._cache_seq_indices = torch.zeros(
            (self._batch_size, self._max_n_beams),
            dtype=torch.long,
            device=self._device,
        )

        self._attention_mask = torch.zeros(
            (self._batch_size, 1, self._max_n_beams, self._max_len),
            dtype=self._dtype,
            device=self._device,
        )

        # Predefined tensors for prefill
        self._predefined_position_ids = torch.arange(
            self._max_len, dtype=torch.long, device=self._device
        ).unsqueeze(0)
        self._predefined_attention_mask = torch.ones(
            (1, 1, self._max_len, self._max_len), dtype=self._dtype, device=self._device
        ).tril_()

        self._kv_prefill_offloading = self._cache_prefill()

        self._logger.debug("Inference controller initialized")

    def _cache_prefill(self):
        # Skip prefill caching if disabled
        if not config.cache_prefill:
            self._logger.info("Prefill caching is disabled - will prefill at runtime")
            return {}

        dataset = util.load_dataset(config.dataset, config.target_model)
        xdg_cache_home = os.environ.get("XDG_CACHE_HOME")

        if xdg_cache_home is None:
            xdg_cache_home = os.path.join(os.path.expanduser("~"), ".cache")

        cache_folder_name = f"{config.target_model}_{config.dataset}"
        cache_dir = Path(xdg_cache_home) / "specedge" / cache_folder_name

        kv_prefill_offloading: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        req_indices = list(range(len(dataset)))
        req_indices = req_indices[config.req_offset :][:: config.sample_req_cnt]

        if not cache_dir.exists():
            cache_dir.mkdir(parents=True, exist_ok=True)

        for req_idx in track(req_indices, description="Prefilling cache"):
            k_cache_file_name = cache_dir / f"{req_idx}_key_cache.pt"
            v_cache_file_name = cache_dir / f"{req_idx}_value_cache.pt"

            if k_cache_file_name.exists() and v_cache_file_name.exists():
                self._logger.debug("Cache files already exist for req_idx=%d", req_idx)
                kv_prefill_offloading[req_idx] = (
                    torch.load(k_cache_file_name, map_location="cpu"),
                    torch.load(v_cache_file_name, map_location="cpu"),
                )
                continue

            prompt = dataset[req_idx]

            self._logger.debug("Creating cache files for req_idx=%d", req_idx)

            input_ids = self._tokenizer.encode(prompt, return_tensors="pt").to(
                self._device
            )[..., :-1]
            position_ids = self._predefined_position_ids[:, : input_ids.size(1)]
            cache_seq_indices = self._predefined_position_ids[:, : input_ids.size(1)]
            attention_mask = self._predefined_attention_mask[
                :, :, : input_ids.size(1), : self._max_len
            ]

            self._engine._past_key_values.clear()

            self._engine.prefill(
                input_ids=input_ids,
                position_ids=position_ids,
                batch_idx=0,
                cache_seq_indices=cache_seq_indices,
                attention_mask=attention_mask,
            )

            k_cache = (
                self._engine._past_key_values.k_cache[
                    :, 0, :, : input_ids.size(-1), ...
                ]
                .squeeze(1)
                .clone()
                .detach()
                .cpu()
            )

            v_cache = (
                self._engine._past_key_values.v_cache[
                    :, 0, :, : input_ids.size(-1), ...
                ]
                .squeeze(1)
                .clone()
                .detach()
                .cpu()
            )

            kv_prefill_offloading[req_idx] = (k_cache, v_cache)

            torch.save(k_cache, k_cache_file_name)
            torch.save(v_cache, v_cache_file_name)

        return kv_prefill_offloading

    def loop(self):
        self._logger.debug("Starting inference loop")
        while True:
            if len(self._request_batches) < self._batch_size:
                while self._check_batch_condition():
                    raw_data = self._recv_queue.get()

                    # Check for sentinel value (shutdown signal)
                    if raw_data is None:
                        self._logger.info("Received shutdown signal in inference loop")
                        self._logger.info(
                            "Processing remaining %d requests before shutdown...",
                            len(self._request_batches),
                        )

                        # Process any remaining requests
                        if len(self._request_batches) > 0:
                            self._logger.info(
                                "Processing final batch of %d requests",
                                len(self._request_batches),
                            )
                            self._client_indices.fill_(-1)

                            with util.Timing(
                                device=self._device, mode="sync"
                            ) as inference_t:
                                forward_t, prefill_indices = self._inference(
                                    self._request_batches[-self._batch_size :]
                                )

                            self._result_logger.log(
                                {
                                    "target": {
                                        "forward_t": forward_t,
                                        "server_end_to_end_t": inference_t.elapsed,
                                        "prefill": len(prefill_indices),
                                    }
                                }
                            )

                        self._logger.info("Inference loop shutting down gracefully")
                        return

                    req = specedge_pb2.ValidateRequest()
                    req.ParseFromString(raw_data)
                    self._request_batches.append(req)

                if len(self._request_batches) == 0:
                    continue

                self._logger.info("Batch size reached: %d", len(self._request_batches))

                self._client_indices.fill_(-1)

                with util.Timing(device=self._device, mode="sync") as inference_t:
                    forward_t, prefill_indices = self._inference(
                        self._request_batches[-self._batch_size :]
                    )
                self._request_batches = self._request_batches[: -self._batch_size]

                self._result_logger.log(
                    {
                        "target": {
                            "forward_t": forward_t,
                            "server_end_to_end_t": inference_t.elapsed,
                            "prefill": len(prefill_indices),
                        }
                    }
                )

    @torch.inference_mode()
    def _inference(self, batch: list[specedge_pb2.ValidateRequest]):
        prefill_indices: list[tuple[int, int]] = []
        self._engine._past_key_values.clear()

        for batch_idx, req in enumerate(batch):
            client_idx = req.client_idx
            self._client_indices[batch_idx] = client_idx

            if req.prefill:
                prefill_indices.append((batch_idx, req.req_idx))
                self._iter_idx[req.client_idx] = 0
            else:
                self._iter_idx[req.client_idx] += 1

            self._input_ids[batch_idx].copy_(
                util.decode(req.input_ids, self._device, torch.long, (-1,))
            )
            self._position_ids[batch_idx].copy_(
                util.decode(req.position_ids, self._device, torch.long, (-1,))
            )
            self._parent_indices[batch_idx].copy_(
                util.decode(req.parent_indices, self._device, torch.long, (-1,))
            )
            self._cache_seq_indices[batch_idx].copy_(
                util.decode(req.cache_seq_indices, self._device, torch.long, (-1,))
            )
            self._attention_mask[batch_idx].copy_(
                util.decode(
                    req.attention_mask,
                    self._device,
                    self._dtype,
                    (1, -1, self._max_len),
                )
            )

            if not req.prefill:
                self._engine._past_key_values.k_cache[:, batch_idx, ...].copy_(
                    self.k_cache[:, req.client_idx, ...]
                )
                self._engine._past_key_values.v_cache[:, batch_idx, ...].copy_(
                    self.v_cache[:, req.client_idx, ...]
                )

        for batch_idx, req_idx in prefill_indices:
            if config.cache_prefill:
                # Load from cache
                k_cache, v_cache = self._kv_prefill_offloading[req_idx]

                self._engine._past_key_values.k_cache[
                    :, batch_idx, :, : k_cache.size(2), :
                ].copy_(k_cache)
                self._engine._past_key_values.v_cache[
                    :, batch_idx, :, : v_cache.size(2), :
                ].copy_(v_cache)
            else:
                # Perform runtime prefill
                req = batch[batch_idx]
                if req.prefix is None or req.prefix == "":
                    raise ValueError(
                        f"Prefix is required for runtime prefill (req_idx={req_idx})"
                    )

                input_ids = self._tokenizer.encode(req.prefix, return_tensors="pt").to(
                    self._device
                )[..., :-1]
                position_ids = self._predefined_position_ids[:, : input_ids.size(1)]
                cache_seq_indices = self._predefined_position_ids[
                    :, : input_ids.size(1)
                ]
                attention_mask = self._predefined_attention_mask[
                    :, :, : input_ids.size(1), : self._max_len
                ]

                self._engine.prefill(
                    input_ids=input_ids,
                    position_ids=position_ids,
                    batch_idx=batch_idx,
                    cache_seq_indices=cache_seq_indices,
                    attention_mask=attention_mask,
                )

        with util.Timing(device=self._device, mode="event") as forward_t:
            logits = self._engine.forward(
                input_ids=self._input_ids,
                position_ids=self._position_ids,
                cache_batch_indices=self._cache_batch_indices.flatten(),
                cache_seq_indices=self._cache_seq_indices.flatten(),
                attention_mask=self._attention_mask,
            )

        selection = util.sampler_from_logits(logits, temperature=self._temperature)
        for batch_idx, client_idx in enumerate(self._client_indices):
            if client_idx == -1:
                continue
            self._resp_queue.put(
                (
                    (util.encode(selection[batch_idx]), len(prefill_indices)),
                    client_idx.item(),
                )
            )

        self._reorder_kv_cache(selection=selection)
        return forward_t.elapsed, prefill_indices

    def _check_batch_condition(self):
        match self._batch_type:
            case "dynamic":
                return (
                    self._recv_queue.qsize() > 0
                    and len(self._request_batches) < self._batch_size
                )
            case "static":
                return len(self._request_batches) < self._batch_size
            case _:
                raise ValueError(f"Unknown batch type: {self._batch_type}")

    def _reorder_kv_cache(self, selection: torch.Tensor):
        offset = self._cache_seq_indices[:, 0][None, :].T

        target_choices_list = []
        for batch_idx in range(self._batch_size):
            offset_b = self._cache_seq_indices[batch_idx, 0]
            parent_indices_b = self._parent_indices[batch_idx] - offset_b
            target_choices_b = selection[batch_idx].flatten()[parent_indices_b]
            target_choices_list.append(target_choices_b)
        target_choices = torch.stack(target_choices_list)

        logit_mask = target_choices == self._input_ids[..., 1:]

        _batch_indices = self._cache_batch_indices.flatten()
        _seq_indices = self._cache_seq_indices.flatten()

        tree_mask = torch.empty(
            (self._batch_size, self._max_budget, self._max_budget),
            dtype=torch.float16,
            device=self._device,
        )

        for batch_idx in range(self._batch_size):
            b_offset = self._cache_seq_indices[batch_idx, 1]
            tree_mask[batch_idx].copy_(
                self._attention_mask[
                    batch_idx, 0, 1:, b_offset : b_offset + self._max_budget
                ]
            )

        position = self._position_ids[:, 1:] - offset

        accepted_mask = logit_mask[:, None, :] & tree_mask.to(torch.bool)

        last_accepted_val, last_accepted_indices = (
            position * (accepted_mask.sum(dim=-1) == position)
        ).max(dim=-1)

        last_accepted = torch.where(
            last_accepted_val == 0, 0, last_accepted_indices + 1
        )

        for batch_idx, client_idx in enumerate(self._client_indices):
            if client_idx == -1:
                continue

            src_mask = self._attention_mask[batch_idx, 0, last_accepted[batch_idx], :]
            b_src_indices = torch.where(src_mask)[0]
            b_dest_indices = torch.arange(
                b_src_indices.size(-1), dtype=torch.long, device=self._device
            )
            self._engine.gather(batch_idx, b_src_indices, b_dest_indices)
            self.k_cache[:, client_idx, ...].copy_(
                self._engine._past_key_values.k_cache[:, batch_idx, ...]
            )
            self.v_cache[:, client_idx, ...].copy_(
                self._engine._past_key_values.v_cache[:, batch_idx, ...]
            )


class DasdInferenceController:
    def __init__(
        self,
        num_clients: int,
        recv_queue: mp.Queue,
        resp_queue: mp.Queue,
    ) -> None:
        self._logger = log.get_logger()
        self._dtype = config.dtype
        self._device = config.device
        self._max_len = config.max_len
        self._num_clients = num_clients

        self._recv_queue = recv_queue
        self._resp_queue = resp_queue

        self._logger.info("Initializing DASD inference controller")
        self._model = util.load_graph_model(
            name=config.target_model,
            device=config.device,
            dtype=config.dtype,
        )
        self._engine = BatchGraphEngine(
            model=self._model,
            max_len=self._max_len,
            max_batch_size=max(1, self._num_clients),
            max_n_beams=1,
            use_cuda_graph=False,
        )
        self._tokenizer = util.load_tokenizer(config.target_model)
        self._dataset = util.load_dataset(config.dataset, config.target_model)
        self._dasd_debug = getattr(config, "dasd_debug", False)
        self._vocab_size = getattr(self._tokenizer, "vocab_size", None)
        if self._vocab_size is None:
            try:
                self._vocab_size = len(self._tokenizer)
            except TypeError:
                self._vocab_size = None

        self._client_slots: dict[str, int] = {}
        self._client_epochs: dict[str, int] = {}
        self._slot_states: dict[str, DasdVerifierState] = {}
        self._terminal_failed_requests: dict[tuple[str, str, int], str] = {}
        self._slot_cap = max(1, self._num_clients)
        self._available_slots = list(range(self._slot_cap))
        self._cuda_poisoned = False
        self._global_poison_reason = ""

    def loop(self):
        self._logger.debug("Starting DASD inference loop")
        while True:
            item = self._recv_queue.get()
            if item is None:
                self._logger.info("Received DASD shutdown signal")
                return

            enqueue_ts, raw_data = item
            if raw_data is None:
                continue

            queue_depth = self._safe_queue_depth()
            request = specedge_pb2.VerifyBundleRequest()
            request.ParseFromString(raw_data)

            process_start = time.perf_counter()
            reject_reason = ""
            verifier_poisoned = self._cuda_poisoned
            try:
                (
                    accept_bitmap,
                    accepted_len,
                    r_obs,
                    reject_reason,
                    verifier_poisoned,
                    verifier_next_token_id,
                    forced_commit_eligible,
                ) = self._verify_bundle(request)
            except Exception as e:
                self._mark_terminal_failed_request(
                    request,
                    reason=f"{type(e).__name__}:{e}",
                )
                if self._is_fatal_cuda_exception(e):
                    self._cuda_poisoned = True
                    self._global_poison_reason = f"{type(e).__name__}:{e}"
                    verifier_poisoned = True
                    reject_reason = f"terminal_failed:{self._global_poison_reason}"
                    state = self._slot_states.get(request.client_id)
                    if state is not None:
                        state.cleanup_safe = False
                        self._client_epochs[request.client_id] = max(
                            self._client_epochs.get(request.client_id, state.epoch),
                            state.epoch,
                        )
                    self._logger.error(
                        "DASD verifier globally poisoned: %s",
                        self._global_poison_reason,
                    )
                    self._dasd_debug_log(
                        "cuda_poisoned",
                        request=request,
                        state=state,
                        reason=type(e).__name__,
                        cuda_mem=self._dasd_cuda_memory(),
                    )
                else:
                    reject_reason = f"terminal_failed:{type(e).__name__}:{e}"
                    self._invalidate_client_state(
                        client_id=request.client_id,
                        reason=f"verification_exception:{type(e).__name__}",
                    )
                self._logger.exception(
                    "DASD verification failed for client=%s request=%s bundle=%s: %s",
                    request.client_id,
                    request.request_id,
                    request.bundle_id,
                    e,
                )
                accept_bitmap = [False] * len(request.token_ids)
                accepted_len = 0
                r_obs = 0.0
                verifier_next_token_id = 0
                forced_commit_eligible = False
            service_ms = (time.perf_counter() - process_start) * 1000.0
            queue_delay_ms = (process_start - enqueue_ts) * 1000.0

            key = (request.client_id, request.request_id, int(request.bundle_id))
            payload = {
                "accept_bitmap": accept_bitmap,
                "accepted_len": accepted_len,
                "r_obs": r_obs,
                "queue_delay_ms": queue_delay_ms,
                "server_service_ms": service_ms,
                "queue_depth": queue_depth,
                "reject_reason": reject_reason,
                "verifier_poisoned": verifier_poisoned,
                "verifier_next_token_id": verifier_next_token_id,
                "forced_commit_eligible": forced_commit_eligible,
            }
            self._resp_queue.put((key, payload))

    def _safe_queue_depth(self):
        try:
            return self._recv_queue.qsize()
        except (NotImplementedError, OSError):
            return -1

    def _terminal_request_key(self, request: specedge_pb2.VerifyBundleRequest):
        return (request.client_id, request.request_id, int(request.epoch))

    def _mark_terminal_failed_request(self, request, reason: str):
        key = self._terminal_request_key(request)
        self._terminal_failed_requests[key] = reason
        self._logger.warning(
            "Marking DASD request terminal-failed client=%s request=%s epoch=%s reason=%s",
            request.client_id,
            request.request_id,
            request.epoch,
            reason,
        )
        self._dasd_debug_log(
            "terminal_failed_request",
            request=request,
            reason=reason,
            cuda_mem=self._dasd_cuda_memory(),
        )

    def _is_fatal_cuda_exception(self, exc: Exception):
        message = f"{type(exc).__name__}: {exc}".lower()
        return "cuda" in message or "device-side assert" in message or "acceleratorerror" in message

    def _dasd_debug_log(
        self,
        event: str,
        request: specedge_pb2.VerifyBundleRequest | None = None,
        state: DasdVerifierState | None = None,
        token_window: list[int] | None = None,
        **extra,
    ):
        if not self._dasd_debug:
            return

        if token_window is None and request is not None:
            token_window = [int(token_id) for token_id in request.token_ids]

        generated_committed_len = None
        prompt_len = None
        slot_idx = None
        state_epoch = None
        if state is not None:
            generated_committed_len = len(state.committed_tokens) - state.prompt_len
            prompt_len = state.prompt_len
            slot_idx = state.slot_idx
            state_epoch = state.epoch
        else:
            prompt_len = extra.get("prompt_len")
            committed_tokens_len = extra.get("committed_tokens_len")
            if prompt_len is not None and committed_tokens_len is not None:
                generated_committed_len = committed_tokens_len - prompt_len

        window_len = len(token_window) if token_window is not None else 0
        token_min = min(token_window) if token_window else None
        token_max = max(token_window) if token_window else None

        self._logger.info(
            "DASD_DEBUG event=%s client_id=%s request_id=%s bundle_id=%s slot_idx=%s "
            "epoch=%s state_epoch=%s base_token_index=%s generated_committed_len=%s prompt_len=%s "
            "window_len=%d token_min=%s token_max=%s %s",
            event,
            request.client_id if request is not None else extra.get("client_id"),
            request.request_id if request is not None else extra.get("request_id"),
            request.bundle_id if request is not None else extra.get("bundle_id"),
            slot_idx if slot_idx is not None else extra.get("slot_idx"),
            request.epoch if request is not None else extra.get("epoch"),
            state_epoch if state_epoch is not None else extra.get("state_epoch"),
            request.base_token_index if request is not None else extra.get("base_token_index"),
            generated_committed_len,
            prompt_len,
            window_len,
            token_min,
            token_max,
            " ".join(f"{key}={value}" for key, value in extra.items()),
        )

    def _dasd_cuda_memory(self):
        if not self._dasd_debug or self._device.type != "cuda" or not torch.cuda.is_available():
            return None

        return {
            "alloc_mb": round(torch.cuda.memory_allocated(self._device) / (1024 * 1024), 1),
            "reserved_mb": round(torch.cuda.memory_reserved(self._device) / (1024 * 1024), 1),
            "max_alloc_mb": round(
                torch.cuda.max_memory_allocated(self._device) / (1024 * 1024), 1
            ),
            "max_reserved_mb": round(
                torch.cuda.max_memory_reserved(self._device) / (1024 * 1024), 1
            ),
        }

    def _safe_reject_bundle(
        self,
        request: specedge_pb2.VerifyBundleRequest,
        token_window: list[int],
        reason: str,
        state: DasdVerifierState | None = None,
        reset_state: bool = False,
    ):
        self._logger.warning(
            "Rejecting DASD bundle client=%s request=%s bundle=%s reason=%s",
            request.client_id,
            request.request_id,
            request.bundle_id,
            reason,
        )
        self._dasd_debug_log(
            "reject_bundle",
            request=request,
            state=state,
            token_window=token_window,
            reason=reason,
            reset_state=reset_state,
        )
        if reset_state:
            self._invalidate_client_state(
                client_id=request.client_id,
                reason=reason,
            )
        return (
            [False] * len(token_window),
            0,
            0.0,
            reason,
            self._cuda_poisoned,
            int(state.next_token_id) if state is not None else 0,
            False,
        )

    def _is_valid_slot_idx(self, slot_idx: int):
        return 0 <= slot_idx < self._slot_cap

    def _invalidate_client_state(self, client_id: str, reason: str):
        state = self._slot_states.pop(client_id, None)
        slot_idx = self._client_slots.get(client_id)
        if state is not None:
            self._client_epochs[client_id] = max(
                self._client_epochs.get(client_id, state.epoch),
                state.epoch,
            )
        self._logger.warning(
            "Invalidating DASD state for client=%s slot_idx=%s reason=%s",
            client_id,
            slot_idx if state is None else state.slot_idx,
            reason,
        )
        self._dasd_debug_log(
            "invalidate_state",
            client_id=client_id,
            request_id=state.request_id if state is not None else None,
            slot_idx=slot_idx if state is None else state.slot_idx,
            epoch=state.epoch if state is not None else self._client_epochs.get(client_id),
            prompt_len=state.prompt_len if state is not None else None,
            reason=reason,
        )

    def _remove_request_slot(
        self,
        slot_idx: int,
        reason: str,
        client_id: str,
        request_id: str,
        epoch: int,
        bundle_id: int | None = None,
    ):
        if self._cuda_poisoned:
            self._dasd_debug_log(
                "remove_requests_skipped_cuda_poisoned",
                client_id=client_id,
                request_id=request_id,
                bundle_id=bundle_id,
                slot_idx=slot_idx,
                epoch=epoch,
                reason=reason,
            )
            return
        self._dasd_debug_log(
            "remove_requests_before",
            client_id=client_id,
            request_id=request_id,
            bundle_id=bundle_id,
            slot_idx=slot_idx,
            epoch=epoch,
            reason=reason,
            cuda_mem=self._dasd_cuda_memory(),
        )
        self._engine.remove_requests(torch.tensor([slot_idx], device=self._device))
        self._dasd_debug_log(
            "remove_requests_after",
            client_id=client_id,
            request_id=request_id,
            bundle_id=bundle_id,
            slot_idx=slot_idx,
            epoch=epoch,
            reason=reason,
            cuda_mem=self._dasd_cuda_memory(),
        )

    def _verify_bundle(self, request: specedge_pb2.VerifyBundleRequest):
        token_window = [int(token_id) for token_id in request.token_ids]
        self._dasd_debug_log("verify_start", request=request, token_window=token_window)

        terminal_reason = self._terminal_failed_requests.get(
            self._terminal_request_key(request)
        )
        if terminal_reason is not None:
            return self._safe_reject_bundle(
                request,
                token_window,
                reason=f"terminal_failed:{terminal_reason}",
            )

        current_epoch = self._client_epochs.get(request.client_id, -1)
        if request.epoch < current_epoch:
            return self._safe_reject_bundle(
                request,
                token_window,
                reason=f"stale_epoch(request={request.epoch}, current={current_epoch})",
            )
        if self._cuda_poisoned:
            return self._safe_reject_bundle(
                request,
                token_window,
                reason=(
                    f"cuda_poisoned:{self._global_poison_reason}"
                    if self._global_poison_reason
                    else "cuda_poisoned"
                ),
            )

        if any(token_id < 0 for token_id in token_window):
            return self._safe_reject_bundle(
                request,
                token_window,
                reason="negative_token_id",
                reset_state=True,
            )
        if self._vocab_size is not None and any(
            token_id >= self._vocab_size for token_id in token_window
        ):
            return self._safe_reject_bundle(
                request,
                token_window,
                reason=f"token_id_out_of_vocab(vocab_size={self._vocab_size})",
                reset_state=True,
            )

        state = self._ensure_client_state(
            client_id=request.client_id,
            request_id=request.request_id,
            epoch=int(request.epoch),
            bundle_id=int(request.bundle_id),
        )
        self._client_epochs[request.client_id] = state.epoch
        if not self._is_valid_slot_idx(state.slot_idx):
            return self._safe_reject_bundle(
                request,
                token_window,
                reason=f"invalid_slot_idx({state.slot_idx})",
                state=state,
                reset_state=True,
            )
        if state.prompt_len > len(state.committed_tokens):
            return self._safe_reject_bundle(
                request,
                token_window,
                reason=(
                    "state_corruption(prompt_len=%d committed_tokens=%d)"
                    % (state.prompt_len, len(state.committed_tokens))
                ),
                state=state,
                reset_state=True,
            )
        self._dasd_debug_log("verify_state_ready", request=request, state=state, token_window=token_window)

        generated_committed_len = len(state.committed_tokens) - state.prompt_len
        if request.base_token_index != generated_committed_len:
            self._logger.debug(
                "Bundle base mismatch for client=%s request=%s epoch=%d bundle=%s (base=%d, committed_generated=%d)",
                request.client_id,
                request.request_id,
                request.epoch,
                request.bundle_id,
                request.base_token_index,
                generated_committed_len,
            )
            self._dasd_debug_log(
                "base_mismatch",
                request=request,
                state=state,
                token_window=token_window,
            )
            return (
                [False] * len(token_window),
                0,
                0.0,
                "base_mismatch",
                False,
                int(state.next_token_id),
                False,
            )

        if request.recovery_fallback_decode:
            fallback_token_id = int(state.next_token_id)
            self._dasd_debug_log(
                "fallback_decode",
                request=request,
                state=state,
                token_window=token_window,
                verifier_next_token_id=fallback_token_id,
            )
            self._advance_state(
                state=state,
                token_id=fallback_token_id,
                bundle_id=int(request.bundle_id),
            )
            return (
                [],
                0,
                1.0,
                "",
                self._cuda_poisoned,
                fallback_token_id,
                True,
            )

        accept_bitmap: list[bool] = []
        accepted_len = 0
        first_expected_token_id = int(state.next_token_id)
        first_proposed_token_id = token_window[0] if token_window else None
        first_mismatch_pos = -1
        if config.dasd_debug:
            self._logger.info(
                "[DASD] verify req=%s bundle=%d epoch=%d base=%d expected_first=%s proposed_first=%s window=%s",
                request.request_id,
                int(request.bundle_id),
                int(request.epoch),
                int(request.base_token_index),
                first_expected_token_id,
                first_proposed_token_id,
                token_window,
            )
        for idx, token_id in enumerate(token_window):
            accepted = int(state.next_token_id) == token_id
            accept_bitmap.append(accepted)
            if not accepted:
                first_mismatch_pos = idx
                if idx < len(token_window) - 1:
                    accept_bitmap.extend([False] * (len(token_window) - idx - 1))
                break

            self._advance_state(
                state=state,
                token_id=token_id,
                bundle_id=int(request.bundle_id),
            )
            accepted_len += 1

        r_obs = accepted_len / len(token_window) if len(token_window) > 0 else 0.0
        self._dasd_debug_log(
            "verify_done",
            request=request,
            state=state,
            token_window=token_window,
            accepted_len=accepted_len,
            r_obs=f"{r_obs:.4f}",
            expected_first_token_id=first_expected_token_id,
            proposed_first_token_id=first_proposed_token_id,
            first_mismatch_pos=first_mismatch_pos,
        )
        if config.dasd_debug:
            self._logger.info(
                "[DASD] verify_result req=%s bundle=%d epoch=%d base=%d accepted=%d/%d mismatch_pos=%d",
                request.request_id,
                int(request.bundle_id),
                int(request.epoch),
                int(request.base_token_index),
                accepted_len,
                len(token_window),
                first_mismatch_pos,
            )
        return (
            accept_bitmap,
            accepted_len,
            r_obs,
            "",
            self._cuda_poisoned,
            first_expected_token_id,
            True,
        )

    def _ensure_client_state(
        self, client_id: str, request_id: str, epoch: int, bundle_id: int
    ):
        state = self._slot_states.get(client_id)

        previous_epoch = self._client_epochs.get(client_id, -1)
        if epoch < previous_epoch:
            raise RuntimeError(
                f"stale epoch for client {client_id}: request epoch {epoch} < current {previous_epoch}"
            )

        if state is not None and state.request_id == request_id:
            previous_committed_len = len(state.committed_tokens) - state.prompt_len
            if epoch < state.epoch:
                raise RuntimeError(
                    f"stale epoch for client {client_id}: request epoch {epoch} < state epoch {state.epoch}"
                )
            if epoch == state.epoch:
                self._dasd_debug_log(
                    "state_reuse",
                    client_id=client_id,
                    request_id=request_id,
                    bundle_id=bundle_id,
                    epoch=epoch,
                    state=state,
                    new_state=False,
                )
                return state

            state.epoch = epoch
            self._client_epochs[client_id] = epoch
            self._dasd_debug_log(
                "epoch_transition_preserve_state",
                client_id=client_id,
                request_id=request_id,
                bundle_id=bundle_id,
                epoch=epoch,
                state=state,
                previous_committed_len=previous_committed_len,
                preserved_committed_len=previous_committed_len,
                new_generated_committed_len=len(state.committed_tokens) - state.prompt_len,
                new_state=False,
            )
            return state

        slot_idx = self._client_slots.get(client_id)
        if slot_idx is None:
            if not self._available_slots:
                raise RuntimeError(
                    "DASD server has no available client slots. "
                    "Increase server.num_clients for DASD mode."
                )
            slot_idx = self._available_slots.pop(0)
            self._client_slots[client_id] = slot_idx
            self._dasd_debug_log(
                "assign_slot",
                client_id=client_id,
                request_id=request_id,
                bundle_id=bundle_id,
                slot_idx=slot_idx,
                epoch=epoch,
                new_state=True,
                reused_slot=False,
            )
        elif not self._is_valid_slot_idx(slot_idx):
            raise RuntimeError(
                f"DASD slot index {slot_idx} for client {client_id} is outside [0, {self._slot_cap - 1}]"
            )
        else:
            self._dasd_debug_log(
                "epoch_or_request_transition",
                client_id=client_id,
                request_id=request_id,
                bundle_id=bundle_id,
                slot_idx=slot_idx,
                epoch=epoch,
                state_epoch=state.epoch if state is not None else previous_epoch,
                prev_request_id=state.request_id if state is not None else None,
                new_state=True,
                reused_slot=True,
            )

        if state is not None and not state.cleanup_safe:
            self._dasd_debug_log(
                "remove_requests_skipped_unsafe_cleanup",
                client_id=client_id,
                request_id=request_id,
                bundle_id=bundle_id,
                slot_idx=slot_idx,
                epoch=epoch,
                state_epoch=state.epoch,
            )
        else:
            self._remove_request_slot(
                slot_idx=slot_idx,
                reason="new_request_switch",
                client_id=client_id,
                request_id=request_id,
                epoch=epoch,
                bundle_id=bundle_id,
            )

        prompt_tokens = self._load_prompt_tokens(request_id)
        self._prime_state_from_prompt(slot_idx, prompt_tokens)

        next_token_id = self._predict_next_token(
            slot_idx=slot_idx,
            token_id=prompt_tokens[-1],
            position=len(prompt_tokens) - 1,
            client_id=client_id,
            request_id=request_id,
            bundle_id=bundle_id,
            epoch=epoch,
            prompt_len=len(prompt_tokens),
            committed_tokens_len=len(prompt_tokens),
            phase="init",
        )
        state = DasdVerifierState(
            client_id=client_id,
            slot_idx=slot_idx,
            request_id=request_id,
            epoch=epoch,
            prompt_len=len(prompt_tokens),
            committed_tokens=prompt_tokens,
            next_token_id=next_token_id,
        )
        self._slot_states[client_id] = state
        self._client_epochs[client_id] = epoch
        self._dasd_debug_log(
            "state_created",
            client_id=client_id,
            request_id=request_id,
            bundle_id=bundle_id,
            epoch=epoch,
            state=state,
            token_window=[],
            new_state=True,
        )
        return state

    def _load_prompt_tokens(self, request_id: str):
        req_idx = self._resolve_req_idx(request_id)
        prompt = self._dataset[req_idx]
        token_ids = self._tokenizer.encode(prompt, return_tensors="pt").to(self._device)[
            0, : self._max_len
        ]
        if token_ids.numel() == 0:
            token_ids = torch.tensor(
                [self._tokenizer.eos_token_id],
                dtype=torch.long,
                device=self._device,
            )
        return token_ids.tolist()

    def _resolve_req_idx(self, request_id: str):
        try:
            req_idx = int(request_id)
        except ValueError as e:
            raise ValueError(
                "VerifyBundleRequest.request_id must be an integer-compatible string "
                "for DASD prompt lookup."
            ) from e

        if req_idx < 0 or req_idx >= len(self._dataset):
            raise ValueError(
                f"request_id {request_id} is out of dataset bounds [0, {len(self._dataset) - 1}]"
            )

        return req_idx

    def _prime_state_from_prompt(self, slot_idx: int, prompt_tokens: list[int]):
        if len(prompt_tokens) <= 1:
            return

        prefix = torch.tensor(prompt_tokens[:-1], dtype=torch.long, device=self._device)
        prefix_len = prefix.numel()

        prefill_input_ids = prefix.unsqueeze(0)
        prefill_position_ids = torch.arange(prefix_len, device=self._device).unsqueeze(0)
        prefill_cache_seq_indices = torch.arange(prefix_len, device=self._device)

        prefill_attention_mask = torch.zeros(
            (1, 1, prefix_len, self._max_len),
            dtype=self._dtype,
            device=self._device,
        )
        prefill_attention_mask[0, 0, :, :prefix_len] = torch.tril(
            torch.ones((prefix_len, prefix_len), dtype=self._dtype, device=self._device)
        )

        self._engine.prefill(
            input_ids=prefill_input_ids,
            position_ids=prefill_position_ids,
            batch_idx=slot_idx,
            cache_seq_indices=prefill_cache_seq_indices,
            attention_mask=prefill_attention_mask,
        )

    def _advance_state(self, state: DasdVerifierState, token_id: int, bundle_id: int):
        position = len(state.committed_tokens)
        next_token_id = self._predict_next_token(
            slot_idx=state.slot_idx,
            token_id=token_id,
            position=position,
            client_id=state.client_id,
            request_id=state.request_id,
            bundle_id=bundle_id,
            epoch=state.epoch,
            prompt_len=state.prompt_len,
            committed_tokens_len=len(state.committed_tokens) + 1,
            phase="advance",
        )
        state.committed_tokens.append(token_id)
        state.next_token_id = next_token_id
        self._dasd_debug_log(
            "advance_state_applied",
            client_id=state.client_id,
            request_id=state.request_id,
            bundle_id=bundle_id,
            slot_idx=state.slot_idx,
            epoch=state.epoch,
            state=state,
            committed_tail=state.committed_tokens[max(0, len(state.committed_tokens) - 8) :],
            next_token_id=state.next_token_id,
            state_tensor_fields="none",
            cleanup_safe=state.cleanup_safe,
            cuda_mem=self._dasd_cuda_memory(),
        )

    @torch.inference_mode()
    def _predict_next_token(
        self,
        slot_idx: int,
        token_id: int,
        position: int,
        client_id: str,
        request_id: str,
        bundle_id: int,
        epoch: int,
        prompt_len: int,
        committed_tokens_len: int,
        phase: str,
    ):
        if position >= self._max_len:
            return self._tokenizer.eos_token_id

        input_ids = torch.tensor([[token_id]], dtype=torch.long, device=self._device)
        position_ids = torch.tensor([[position]], dtype=torch.long, device=self._device)
        cache_batch_indices = torch.zeros((1,), dtype=torch.long, device=self._device)
        cache_seq_indices = torch.tensor([position], dtype=torch.long, device=self._device)
        attention_mask = torch.zeros(
            (1, 1, 1, self._max_len), dtype=self._dtype, device=self._device
        )
        attention_mask[0, 0, 0, : position + 1] = 1

        if (
            input_ids.numel() != 1
            or position_ids.numel() != 1
            or cache_batch_indices.numel() != 1
            or cache_seq_indices.numel() != 1
        ):
            self._dasd_debug_log(
                "predict_shape_invalid",
                client_id=client_id,
                request_id=request_id,
                bundle_id=bundle_id,
                epoch=epoch,
                slot_idx=slot_idx,
                prompt_len=prompt_len,
                committed_tokens_len=committed_tokens_len,
                input_shape=tuple(input_ids.shape),
                input_len=input_ids.size(-1),
                attention_mask_shape=tuple(attention_mask.shape),
                position_shape=tuple(position_ids.shape),
                position_len=position_ids.size(-1),
                cache_batch_indices_shape=tuple(cache_batch_indices.shape),
                cache_batch_indices=cache_batch_indices.tolist(),
                cache_seq_indices_shape=tuple(cache_seq_indices.shape),
                cache_seq_indices=cache_seq_indices.tolist(),
                decode_tokens=input_ids.numel(),
                single_token_prediction=True,
                phase=phase,
            )
            raise RuntimeError(
                "DASD single-token prediction expected exactly one token/position/batch index"
            )

        self._dasd_debug_log(
            "predict_step",
            client_id=client_id,
            request_id=request_id,
            bundle_id=bundle_id,
            epoch=epoch,
            slot_idx=slot_idx,
            prompt_len=prompt_len,
            committed_tokens_len=committed_tokens_len,
            input_shape=tuple(input_ids.shape),
            input_len=input_ids.size(-1),
            attention_mask_shape=tuple(attention_mask.shape),
            position_shape=tuple(position_ids.shape),
            position_len=position_ids.size(-1),
            cache_batch_indices_shape=tuple(cache_batch_indices.shape),
            cache_batch_indices=cache_batch_indices.tolist(),
            cache_seq_indices_shape=tuple(cache_seq_indices.shape),
            cache_seq_indices=cache_seq_indices.tolist(),
            decode_tokens=input_ids.numel(),
            single_token_prediction=True,
            phase=phase,
            cuda_mem=self._dasd_cuda_memory(),
            grad_enabled=torch.is_grad_enabled(),
        )

        # DASD verifier decode should reuse the existing slot KV cache in place.
        # prefill_context() clones the full slot cache and causes per-step memory growth.
        with self._engine._past_key_values.slot_view_context(1, slot_idx):
            logits, returned_past_key_values = self._model.forward(
                input_ids=input_ids,
                position_ids=position_ids,
                cache_batch_indices=cache_batch_indices,
                cache_seq_indices=cache_seq_indices,
                attention_mask=attention_mask,
                past_key_values=self._engine._past_key_values,
            )
            self._dasd_debug_log(
                "predict_forward_done",
                client_id=client_id,
                request_id=request_id,
                bundle_id=bundle_id,
                epoch=epoch,
                slot_idx=slot_idx,
                prompt_len=prompt_len,
                committed_tokens_len=committed_tokens_len,
                phase=phase,
                logits_shape=tuple(logits.shape),
                logits_requires_grad=bool(logits.requires_grad),
                returned_cache_is_engine_cache=(
                    returned_past_key_values is self._engine._past_key_values
                ),
                returned_cache_type=type(returned_past_key_values).__name__,
                state_tensor_fields="none",
                cuda_mem=self._dasd_cuda_memory(),
            )
            next_token_id = int(logits[0, 0].argmax(dim=-1).item())
            del logits
            del returned_past_key_values
        self._dasd_debug_log(
            "predict_step_done",
            client_id=client_id,
            request_id=request_id,
            bundle_id=bundle_id,
            epoch=epoch,
            slot_idx=slot_idx,
            prompt_len=prompt_len,
            committed_tokens_len=committed_tokens_len,
            cache_batch_indices_shape=tuple(cache_batch_indices.shape),
            cache_batch_indices=cache_batch_indices.tolist(),
            cache_seq_indices_shape=tuple(cache_seq_indices.shape),
            cache_seq_indices=cache_seq_indices.tolist(),
            input_shape=tuple(input_ids.shape),
            input_len=input_ids.size(-1),
            position_shape=tuple(position_ids.shape),
            position_len=position_ids.size(-1),
            decode_tokens=input_ids.numel(),
            single_token_prediction=True,
            phase=phase,
            cuda_mem=self._dasd_cuda_memory(),
            grad_enabled=torch.is_grad_enabled(),
        )
        return next_token_id


def _init_inference(
    num_clients: int,
    recv_queue: mp.Queue,
    resp_queue: mp.Queue,
):
    # Configure logging in child process
    from config import SpecEdgeBatchServerConfig as config

    log_config = log.get_default_log_config(
        Path(config.result_path) / config.exp_name, "server"
    )
    log.configure_logging(log_config)

    try:
        controller = InferenceController(num_clients, recv_queue, resp_queue)
        controller.loop()
    except KeyboardInterrupt:
        # Gracefully exit without printing traceback
        pass


def _init_dasd_inference(
    num_clients: int,
    recv_queue: mp.Queue,
    resp_queue: mp.Queue,
):
    from config import SpecEdgeBatchServerConfig as config

    log_config = log.get_default_log_config(
        Path(config.result_path) / config.exp_name, "server"
    )
    log.configure_logging(log_config)

    try:
        controller = DasdInferenceController(num_clients, recv_queue, resp_queue)
        controller.loop()
    except KeyboardInterrupt:
        pass
