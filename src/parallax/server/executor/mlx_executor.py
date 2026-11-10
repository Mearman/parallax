"""
MLX-LM backend implementation of high level executor
"""

import logging
import pickle
import time
from typing import Any, Dict, List, Optional, Tuple

import mlx.core as mx
import numpy as np

from parallax.server.cache_manager import CacheManager
from parallax.server.executor.base_executor import BaseExecutor
from parallax.server.inline_prefill import continuation_inputs, prefill_in_slices
from parallax.server.request import (
    InitialRequest,
    IntermediateRequest,
    Request,
    RequestStatus,
)
from parallax.server.sampling.sampler import SamplingBatchInfo
from parallax.server.speculation import StepTimings, accepted_token_ids, commit_accepted
from parallax.server.split_speculation import FirstPeerSpeculation, Step
from parallax.server.shard_loader import MLXModelLoader
from parallax.utils.chunked_prefill import (
    complete_local_middle_chunk,
    filter_middle_chunk_next_batch,
)
from parallax.utils.layer_types import INDEX_CACHE_LAYER_TYPES, MLA_CACHE_LAYER_TYPES
from parallax.utils.mac_prefill_adder import AddReqResult, MACPrefillAdder
from parallax.utils.utils import (
    combine_padding_and_causal_masks,
    create_causal_mask,
    get_device_dtype,
    get_layer_types,
    pad_inputs,
)
from parallax_utils.logging_config import get_logger

logger = get_logger(__name__)


class MLXExecutor(BaseExecutor):
    def __init__(
        self,
        # Model Configs
        model_repo: str,
        start_layer: int,
        end_layer: int,
        dtype: str = "float16",
        # Device override
        device: Optional[str] = None,
        use_hfcache: bool = False,
        # Scheduler Configs
        max_batch_size: Optional[int] = 8,
        max_sequence_length: Optional[int] = None,
        max_tokens_in_kv_pool: Optional[int] = None,
        # Controlling perfill / decode ratio
        max_num_tokens_per_batch: int = 16384,
        prefill_priority: int = 0,
        micro_batch_ratio: int = 2,
        scheduler_wait_ms: int = 500,
        request_timeout_s: Optional[int] = 600,
        # Metrics Configs
        layer_latency_update_every: int = 4096,
        # KV Cache Configs
        kv_block_size: int = 32,
        kv_cache_memory_fraction: float = 0.8,
        enable_prefix_cache: Optional[bool] = False,
        # Communication Configs
        # P2P Communication Configs
        send_to_peer_addr: Optional[str] = None,
        recv_from_peer_addr: Optional[str] = None,
        # IPC Communication Configs
        executor_input_ipc_addr: Optional[str] = None,
        executor_output_ipc_addr: Optional[str] = None,
        # GPU Specialized Configs
        attention_backend: Optional[str] = "flashinfer",
        moe_runner_backend: Optional[str] = "auto",
        enable_lora: Optional[bool] = False,
        max_lora_rank: Optional[int] = None,
        lora_target_modules: Optional[List[str]] = None,
        lora_paths: Optional[List[str]] = None,
        max_loras_per_batch: Optional[int] = None,
        max_loaded_loras: Optional[int] = None,
        lora_eviction_policy: Optional[str] = "lru",
        lora_backend: Optional[str] = "triton",
        max_lora_chunk_size: Optional[int] = 128,
        # Tensor Parallel Configs
        tp_rank: Optional[int] = 0,
        tp_size: Optional[int] = 1,
        nccl_port: Optional[int] = 4000,
        # Data Parallel Configs (not used in MLX, but accepted for compatibility)
        enable_dp_attention: Optional[bool] = False,
        dp_rank: Optional[int] = 0,
        dp_size: Optional[int] = 1,
        # Optional shared state for layer reallocation detection (when running in subprocess)
        shared_state: Optional[dict] = None,
        # Weight Refit
        enable_weight_refit: Optional[bool] = False,
        weight_refit_mode: Optional[str] = "disk",
        chunked_prefill_size: Optional[int] = None,
        # First peer without a prefix cache: prefill a prompt in slices of this many tokens (0 prefills it whole)
        inline_prefill_chunk_size: int = 0,
        # Pipe communication
        conn: Optional[List[Any]] = [],
    ):
        group = mx.distributed.init()
        tp_size = group.size()
        tp_rank = group.rank()

        logger.debug(
            f"Initializing MLX sharded model loader for repo={model_repo}, layers=[{start_layer}, {end_layer}), tp_rank={tp_rank}, tp_size={tp_size}"
        )

        try:
            mx.set_wired_limit(mx.device_info()["max_recommended_working_set_size"])
        except Exception:
            logger.warning("Using mlx without metal backend.")

        self.shard_loader = MLXModelLoader(
            model_repo,
            start_layer=start_layer,
            end_layer=end_layer,
            use_hfcache=use_hfcache,
        )
        t0 = time.time()
        self.model_shard, self.config, self.tokenizer = self.shard_loader.load()

        adapters = lora_paths[0] if lora_paths else None
        if adapters:
            logger.debug(f"mlx adapters is: {adapters}")
            self.model_shard = self.shard_loader.load_lora(self.model_shard, adapters)

        logger.debug(
            f"MLX sharded model loaded in {(time.time() - t0) * 1000:.1f} ms; num_layers={self.config.get('num_hidden_layers')}"
        )

        # TODO: Duplicate code to BaseExecutor since num_shard_layers and dtype are needed for initializing kv cache
        self.num_shard_layers = end_layer - start_layer
        self.dtype = get_device_dtype(dtype, device)
        logger.debug(
            f"Executor dtype set to {dtype} (resolved={self.dtype}); shard_layers={self.num_shard_layers}"
        )

        # Calculate feature dimensions for kv cache
        num_key_value_heads = self.config.get("num_key_value_heads")
        if num_key_value_heads is None:
            # Step3.5 flash use num_attention_groups instead.
            num_key_value_heads = self.config.get("num_attention_groups")
        head_dim = self.config.get("head_dim") or self.config.get("hidden_size") // self.config.get(
            "num_attention_heads"
        )
        qk_nope_head_dim = self.config.get("qk_nope_head_dim", None)
        qk_rope_head_dim = self.config.get("qk_rope_head_dim", None)
        if qk_nope_head_dim is not None and qk_rope_head_dim is not None:
            logger.debug(
                f"qk_nope_head_dim={qk_nope_head_dim}, qk_rope_head_dim={qk_rope_head_dim}"
            )
            head_dim = qk_nope_head_dim + qk_rope_head_dim

        v_head_dim = self.config.get("v_head_dim", None)
        linear_key_head_dim = self.config.get("linear_key_head_dim", None)
        linear_value_head_dim = self.config.get("linear_value_head_dim", None)
        linear_conv_kernel_dim = self.config.get("linear_conv_kernel_dim", None)
        linear_num_key_heads = self.config.get("linear_num_key_heads", None)
        linear_num_value_heads = self.config.get("linear_num_value_heads", None)
        key_dim, value_dim, conv_dim = None, None, None
        if linear_key_head_dim is not None and linear_num_key_heads is not None:
            key_dim = linear_key_head_dim * linear_num_key_heads
        if linear_value_head_dim is not None and linear_num_value_heads is not None:
            value_dim = linear_value_head_dim * linear_num_value_heads
        if key_dim is not None and value_dim is not None:
            conv_dim = key_dim * 2 + value_dim

        index_head_dim = self.config.get("index_head_dim", None)
        index_n_heads = self.config.get("index_n_heads", None)
        kv_lora_rank = self.config.get("kv_lora_rank", None)

        layer_types = get_layer_types(self.config, start_layer, end_layer)
        uses_mla_cache = any(t in MLA_CACHE_LAYER_TYPES for t in layer_types)
        uses_index_cache = any(t in INDEX_CACHE_LAYER_TYPES for t in layer_types)
        index_key_heads = 1 if uses_index_cache else None
        sliding_window = self.config.get("sliding_window", None)
        use_sliding_window = self.config.get("use_sliding_window", None)
        if use_sliding_window is False:
            sliding_window = None

        # Validate and adjust block size for Metal backend
        supported_block_sizes = [1, 8, 16, 32, 64]
        if kv_block_size not in supported_block_sizes:
            nearest_block_size = min(supported_block_sizes, key=lambda x: abs(x - kv_block_size))
            logger.warning(
                f"Block size {kv_block_size} is not supported for MLX Metal backend. "
                f"Supported block sizes are {supported_block_sizes}. "
                f"Automatically adjusting to supported block size: {nearest_block_size}"
            )
            kv_block_size = nearest_block_size

        logger.debug(
            "Initializing CacheManager (mlx) with block_size=%d, layers=%d",
            kv_block_size,
            self.num_shard_layers,
        )
        normalized_chunked_prefill_size = None
        if chunked_prefill_size is not None:
            if chunked_prefill_size < 0:
                raise ValueError("chunked_prefill_size must be non-negative")
            if chunked_prefill_size == 0:
                chunked_prefill_size = None
            elif not enable_prefix_cache:
                raise ValueError(
                    "Chunked prefill is not supported when disable prefix cache, please enable prefix cache or set --chunked-prefill-size 0"
                )
            else:
                normalized_chunked_prefill_size = (
                    (chunked_prefill_size + kv_block_size - 1) // kv_block_size * kv_block_size
                )

        self.cache_manager = CacheManager(
            num_layers=self.num_shard_layers,
            num_kv_heads=num_key_value_heads // tp_size,
            head_dim=head_dim,
            dtype=self.dtype,
            block_size=kv_block_size,
            cache_memory_fraction=kv_cache_memory_fraction,
            head_dim_v=v_head_dim,
            index_head_dim=index_head_dim if uses_index_cache else None,
            index_n_heads=index_n_heads if uses_index_cache else None,
            index_key_heads=index_key_heads,
            kv_lora_rank=kv_lora_rank if uses_mla_cache else None,
            qk_rope_head_dim=qk_rope_head_dim if uses_mla_cache else None,
            layer_types=layer_types,
            max_num_seqs=max_batch_size // micro_batch_ratio,
            conv_dim=conv_dim,
            conv_kernel_size=linear_conv_kernel_dim,
            linear_k_dim=linear_key_head_dim,
            linear_v_dim=linear_value_head_dim,
            linear_num_k_heads=linear_num_key_heads,
            linear_num_v_heads=linear_num_value_heads,
            enable_prefix_cache=enable_prefix_cache,
            sliding_window=sliding_window,
            chunked_prefill_size=normalized_chunked_prefill_size,
        )

        self.chunked_prefill_size = normalized_chunked_prefill_size
        self.cache_manager.chunked_prefill_size = self.chunked_prefill_size
        self.cache_manager.defer_prefill_allocation = self.chunked_prefill_size is not None

        super().__init__(
            start_layer=start_layer,
            end_layer=end_layer,
            dtype=dtype,
            device=device,
            max_batch_size=max_batch_size,
            max_sequence_length=max_sequence_length,
            max_num_tokens_per_batch=max_num_tokens_per_batch,
            prefill_priority=prefill_priority,
            micro_batch_ratio=micro_batch_ratio,
            scheduler_wait_ms=scheduler_wait_ms,
            request_timeout_s=request_timeout_s,
            layer_latency_update_every=layer_latency_update_every,
            send_to_peer_addr=send_to_peer_addr,
            recv_from_peer_addr=recv_from_peer_addr,
            executor_input_ipc_addr=executor_input_ipc_addr,
            executor_output_ipc_addr=executor_output_ipc_addr,
            tp_rank=tp_rank,
            tp_size=tp_size,
            shared_state=shared_state,
            enable_weight_refit=enable_weight_refit,
            weight_refit_mode=weight_refit_mode,
            chunked_prefill_size=self.chunked_prefill_size,
            kv_block_size=kv_block_size,
            conn=conn,
        )

        # Prefix Cache Manager
        self.enable_prefix_cache = enable_prefix_cache
        self.chunked_req = None
        # Speculation across a split: the first peer folds the drafts the last peer proposes into its decode
        # steps. It is inert until a returned token carries drafts.
        self.speculation = FirstPeerSpeculation(self.cache_manager) if self.is_first_peer else None
        self._verify_steps: Dict[str, Step] = {}
        self.step_timings = StepTimings()
        # When each request's verify step was sent down the pipeline.
        self._sent_at: Dict[str, float] = {}
        if inline_prefill_chunk_size and (enable_prefix_cache or not self.is_first_peer):
            raise ValueError(
                "--inline-prefill-chunk-size is for a first peer without a prefix cache; "
                "a peer with a prefix cache chunks through it (--chunked-prefill-size)"
            )
        self.inline_prefill_chunk_size = inline_prefill_chunk_size
        # self.prefix_cache = RadixCache(
        #     num_kv_heads=num_key_value_heads,
        #     head_dim=head_dim,
        #     head_dim_v=v_head_dim,
        #     num_layers=self.num_shard_layers,
        #     dtype=self.dtype,
        #     page_size=1,
        # )
        logger.debug(
            f"mlx_executor initialized; wired_limit set; prefix_cache={'on' if self.enable_prefix_cache else 'off'}, total memory usage: {mx.get_active_memory() / 1024**3 :.3f} GB"
        )

    def _tensor_parallel_broadcast_pyobj(self, broadcast_obj):
        """Wrapper for broadcast pyobject in TP group using send/recv with explicit sync"""
        if self.tp_size <= 1:
            return broadcast_obj

        data_len = 0
        data = None
        if self.tp_rank == 0:
            # Rank 0 prepares data
            data = pickle.dumps(broadcast_obj)
            data_len = len(data)

        # Broadcast length using all_sum
        data_len_arr = mx.array([data_len], dtype=mx.int32)
        data_len_arr = mx.distributed.all_sum(data_len_arr)
        data_len = int(data_len_arr[0])

        if data is not None:
            data_arr = mx.array(np.frombuffer(data, dtype=np.uint8))
        else:
            data_arr = mx.zeros((data_len,), dtype=mx.uint8)

        data_arr = mx.distributed.all_sum(data_arr)
        data = pickle.loads(np.array(data_arr).tobytes())
        return data

    def _complete_local_middle_chunk(self, request_id: str):
        """Finish local bookkeeping after this peer runs a non-final prefill chunk."""
        complete_local_middle_chunk(
            self,
            request_id,
            release_callback=self.cache_manager.release_request,
        )

    def handle_input_requests(self, requests: List[Request]):
        """Update requests states and status in scheduler and cache manager."""
        mx.eval(mx.zeros(1024))
        if self.tp_size > 1:
            requests = self._tensor_parallel_broadcast_pyobj(requests)

        if len(requests) == 0:
            return

        if self.is_first_peer:
            # First peer can receive InitialRequests from the client RPC,
            # or IntermediateRequests from the last peer.
            for req in requests:
                if isinstance(req, InitialRequest):
                    self.scheduler.enque_request(req)
                elif isinstance(req, IntermediateRequest):
                    original_req = self.scheduler.get_running_request(req.request_id)
                    if original_req is None:
                        logger.warning(
                            f"Received IntermediateRequest {req.request_id}. "
                            "But no corresponding request found in scheduler. "
                            "It might have been cancelled or finished."
                        )
                        continue
                    if not self.cache_manager.has_request(req.request_id):
                        logger.warning(
                            f"Received IntermediateRequest {req.request_id}. "
                            "But no corresponding request found in cache manager. "
                            "It might have been cancelled or finished."
                        )
                        continue

                    committed: List[int] = []
                    if not req.abort and req.next_token_id is not None:
                        committed = self._commit_returned_tokens(original_req, req)

                    if len(req.routing_table) > 0:
                        original_req.routing_table = req.routing_table

                    # Check for termination.
                    if req.abort:
                        original_req.abort = True

                    if self.scheduler.check_and_update_request_status(original_req):
                        timing = self.step_timings.summary(original_req.request_id)
                        if timing:
                            logger.info(timing)
                        self.speculation.release(original_req.request_id)
                        self.cache_manager.release_request(original_req.request_id)
                        logger.debug(
                            f"Released resources for finished request {req.request_id}, "
                            f"memory usage: {mx.get_active_memory() / 1024**3 :.3f} GB"
                        )
                        if not self.is_last_peer and not req.abort:
                            self.finished_batch.append(req)
                    else:
                        self.scheduler.enque_request(original_req)

                    # Send token/terminal update to the Rust frontend: every committed token, the last one
                    # carrying the request's finish state.
                    if self.tp_rank == 0:
                        for token_id in committed[:-1]:
                            self._send_engine_core_token_output(
                                request_id=original_req.request_id,
                                token_id=token_id,
                                finish_reason=None,
                                stop_reason=None,
                            )
                        self.send_engine_core_request_output(
                            request=original_req,
                            token_id=committed[-1] if committed else req.next_token_id,
                        )
                else:
                    raise TypeError(f"First peer received unexpected request type: {type(req)}")

        else:
            # Intermediate and Last peers receive IntermediateRequests from the previous peer.
            for req in requests:
                assert isinstance(
                    req, IntermediateRequest
                ), "Non-first peers must receive IntermediateRequests."
                if req.is_finished or req.hidden_states is None:
                    self.cache_manager.release_request(req.request_id)
                    logger.debug(
                        f"Released resources for finished request {req.request_id}, "
                        f"memory usage: {mx.get_active_memory() / 1024**3 :.3f} GB"
                    )
                    self.scheduler.evict_request(req.request_id)
                    if not self.is_last_peer and not req.abort:
                        self.finished_batch.append(req)
                else:
                    # This is an active request, add it to the scheduler queue to be processed.
                    self.scheduler.enque_request(req)

    def _commit_returned_tokens(self, original_req: InitialRequest, returned: IntermediateRequest) -> List[int]:
        """Commit the tokens the last peer returned for a request and tell the speculation what it accepted.

        A step returns the drafts it accepted followed by the token it sampled, and the next step's drafts. The first
        return after the prompt starts speculation if it carries drafts. A request that finishes inside the accepted
        tokens ends the step there, so tokens beyond it are never committed.
        """
        accepted = accepted_token_ids(returned)
        drafts = returned.draft_token_ids or []
        rid = original_req.request_id
        arrived = self.step_timings.clock()
        if rid in self._sent_at:
            self.step_timings.record(rid, "round_trip", arrived - self._sent_at.pop(rid))
        if self.speculation.in_flight(rid):
            self.speculation.finish(rid, accepted, drafts)
        elif drafts:
            self.speculation.start(rid, drafts)
        committed = commit_accepted(original_req, accepted, self.scheduler.check_and_update_request_status)
        self.step_timings.record(rid, "apply_verdict", self.step_timings.clock() - arrived)
        return committed

    def prepare_next_batch_requests(
        self, requests: List[Request], batch_output: Any, context_lengths: Any
    ) -> List[Request]:
        steps = batch_output.get("verify_steps") if isinstance(batch_output, dict) else None
        if steps is not None:
            return self._prepare_next_verify_requests(requests, batch_output["hidden_states"], steps)
        next_batch = super().prepare_next_batch_requests(requests, batch_output, context_lengths)
        return filter_middle_chunk_next_batch(
            self,
            requests,
            next_batch,
            release_callback=self.cache_manager.release_request,
        )

    def _prepare_next_verify_requests(
        self, requests: List[Request], hidden_states: mx.array, steps: List[Step]
    ) -> List[Request]:
        """The requests for the next peer after verify steps: the rows of the input token and its drafts, with the drafts."""
        next_batch = []
        for index, (request, step) in enumerate(zip(requests, steps, strict=True)):
            rows = hidden_states[index, len(step.tokens) - step.send_rows : len(step.tokens), :]
            next_request = self._prepare_next_single_request(request, rows, None)
            next_request.draft_token_ids = list(self.speculation.drafts(request.request_id))
            self._sent_at[request.request_id] = self.step_timings.clock()
            next_batch.append(next_request)
        return next_batch

    def process_batch(self, prepared_inputs: Dict[str, Any], return_decoded_tokens: bool = True):
        """Process a batch of requests in MLX."""
        # Run model and get updated cache
        # Note: Paged Attention writes KV cache in-place within the model (via reshape_and_cache).
        # The returned 'hidden_states' is what we need.
        # The returned cache tuple (_, _) is ignored/unused here.

        # Check if this is a prefill batch
        is_prefill_batch = any(req.is_prefill for req in prepared_inputs["requests"])

        start_time = time.time()
        if "inline_prompts" in prepared_inputs:
            hidden_states = self._prefill_inline(prepared_inputs)
        else:
            hidden_states = self.model_shard(
                h_or_tokens=prepared_inputs["h_or_tokens"],
                cache=prepared_inputs["cache"],
                mask=prepared_inputs.get("mask"),
                block_tables=prepared_inputs.get("block_tables"),
                context_lengths=prepared_inputs.get("context_lengths"),
                slot_mapping=prepared_inputs.get("slot_mapping"),
                state_slot_mapping=prepared_inputs.get("state_slot_mapping"),
                prefix_lens=prepared_inputs.get("prefix_lens"),  # For RoPE offset in prefix cache
            )
        mx.eval(hidden_states)

        if self.cache_manager.needs_slots:
            self.cache_manager.materialize_linear_caches()
        if prepared_inputs.get("verify_steps"):
            chunk_seconds = time.time() - start_time
            for req in prepared_inputs["requests"]:
                self.step_timings.record(req.request_id, "m3_chunk", chunk_seconds)

        if logger.isEnabledFor(logging.DEBUG):
            forward_time = (time.time() - start_time) * 1000

            if is_prefill_batch:
                # Log prefill time and prefix cache hit info
                total_tokens = sum(
                    prepared_inputs.get(
                        "actual_processed_lengths",
                        [req.total_length for req in prepared_inputs["requests"]],
                    )[i]
                    for i, req in enumerate(prepared_inputs["requests"])
                    if req.is_prefill
                )
                matched_tokens = sum(
                    prepared_inputs.get("prefix_lens", [0] * len(prepared_inputs["requests"]))[i]
                    for i, req in enumerate(prepared_inputs["requests"])
                    if req.is_prefill
                )
                logger.info(
                    f"[PREFILL] time: {forward_time:.3f} ms, "
                    f"tokens: {total_tokens}, prefix_cached: {matched_tokens}"
                )
            else:
                logger.debug(f"model forward pass done, time: {forward_time:.3f} ms")

        logger.debug(
            f"Processed batch of {len(prepared_inputs['requests'])} requests, "
            f"request status: {prepared_inputs['requests'][0].status}, "
            f"hidden_states shape: {hidden_states.shape}"
        )

        lengths = mx.zeros((len(prepared_inputs["requests"]),), dtype=mx.int32)
        requests = prepared_inputs["requests"]
        for i, req in enumerate(requests):
            if req.is_prefill:
                # Use actual_processed_lengths if available (for prefix cache case),
                # otherwise use context_lengths (total_length)
                if "actual_processed_lengths" in prepared_inputs:
                    lengths[i] = prepared_inputs.get("actual_processed_lengths")[i]
                else:
                    lengths[i] = prepared_inputs.get("context_lengths")[i]
            elif req.is_decoding:
                lengths[i] = 1
            else:
                continue

        # Note: With PagedAttention, we don't need to explicitly update requests with new K/V
        # because they are written in-place to the global cache.
        # self.cache_manager.update_requests(...) is REMOVED.

        # Update prefix cache: insert full blocks after prefill
        if self.enable_prefix_cache:
            if is_prefill_batch:
                mx.synchronize()
            for req in requests:
                if req.is_prefill:
                    # Insert all full blocks from this prefill into the prefix cache
                    self.cache_manager.insert_full_blocks_to_cache(req.request_id)

        # Process last peer: need additional sampling + detokenization
        if return_decoded_tokens:
            sampling_info = SamplingBatchInfo.from_reqs(requests)

            # For MLX, hidden_states at last shard is already logits (after lm_head)
            # hidden_states shape: [batch_size, seq_len, vocab_size]
            token_ids = mx.array(
                self.model_shard.logits_to_tokens(hidden_states, lengths, sampling_info)
            )

            needs_probs = any(
                (isinstance(req, InitialRequest) and req.return_probs)
                or (isinstance(req, IntermediateRequest) and req.return_probs)
                for req in requests
            )

            token_probs = None
            if needs_probs:
                # Extract probability values for sampled tokens
                try:
                    # Get last position logits for each request
                    batch_probs = []
                    for i, req in enumerate(requests):
                        if lengths[i] > 0:
                            # Get logit at last position
                            last_idx = int(lengths[i]) - 1
                            last_logits = hidden_states[i, last_idx, :]  # [vocab_size]
                            probs = last_logits / sampling_info.temperatures.reshape(-1, 1)
                            probs[:] = mx.softmax(probs, axis=-1)
                            # logit_value = float(last_logits[token_id])
                            # batch_logits.append(logit_value)
                            # Extract probability for the sampled token
                            token_id = int(token_ids[i])
                            batch_probs.append(float(probs[i, token_id]))

                    token_probs = batch_probs if batch_probs else None
                except Exception as e:
                    logger.debug(f"Failed to extract token probs: {e}")
                    token_probs = None

            # Return dict with token_ids and optional probs
            return {"hidden_states": token_ids, "probs": token_probs}
        # Intermediate peer: return hidden states without probs
        return {
            "hidden_states": hidden_states,
            "probs": None,
            "verify_steps": prepared_inputs.get("verify_steps"),
        }

    def _release_request(self, rid: str):
        """Release per-request resources in MLX."""
        try:
            if getattr(self, "speculation", None) is not None:
                self.speculation.release(rid)
                self._verify_steps.pop(rid, None)
            if hasattr(self, "cache_manager") and self.cache_manager is not None:
                self.cache_manager.release_request(rid)
        except Exception:
            pass

    def _gen_token_id_from_hidden(self, hidden_states) -> Tuple[int, Any]:
        """
        Inplace modifies hidden_states.
        Returns token_id, hidden_states
        """
        assert hidden_states.dtype == mx.uint32, "Single node must generate an output_id."
        next_token_id = int(hidden_states[0])
        hidden_states = hidden_states.astype(mx.int32)
        return next_token_id, hidden_states

    def _prepare_prefill_batch(self, batched_requests: List[Request]) -> Dict[str, Any]:
        """Prepares inputs for ShardedModel from a batch of prefill requests."""
        if self.inline_prefill_chunk_size:
            return self._prepare_inline_prefill_batch(batched_requests)
        batch_size = len(batched_requests)
        if batch_size == 0:
            return None

        if self.chunked_prefill_size is not None:
            original_batched_requests = batched_requests
            adder = MACPrefillAdder(
                self.cache_manager.block_size, self.chunked_prefill_size, self.cache_manager
            )
            chunked_rid = self.chunked_req.rid if self.chunked_req is not None else None

            if self.chunked_req is not None and chunked_rid in [
                req.request_id for req in original_batched_requests
            ]:
                self.chunked_req = [
                    req for req in original_batched_requests if req.request_id == chunked_rid
                ][0]
                self.chunked_req = adder.add_chunked_req(self.chunked_req)

            for old_req in original_batched_requests:
                if chunked_rid is not None and old_req.request_id == chunked_rid:
                    continue
                res = adder.add_one_req(old_req)
                if res != AddReqResult.CONTINUE:
                    break

            if adder.new_chunked_req is not None:
                self.chunked_req = adder.new_chunked_req

            if self.chunked_req is not None and self.chunked_req.rid in [
                req.request_id for req in original_batched_requests
            ]:
                self.chunked_req.is_chunked = True

            can_run_by_id = {req.request_id: req for req in adder.can_run_list}
            batched_requests = [
                can_run_by_id[req.request_id]
                for req in original_batched_requests
                if req.request_id in can_run_by_id
            ]
            batch_size = len(batched_requests)
            if batch_size == 0:
                return None

        h_or_tokens_list = []
        block_tables_list = []
        context_lengths_list = []
        prefix_lens_list = []  # Track matched prefix lengths for each request
        actual_processed_lengths_list = []  # Track actual processed token lengths for each request

        for req in batched_requests:
            assert req.is_prefill, f"Request {req.request_id} is not a prefill request."

            # Allocate Paged KV blocks with prefix cache support
            # All peers can perform prefix cache matching if input_ids are available
            token_ids = None
            if self.enable_prefix_cache and req.input_ids is not None:
                token_ids = req.input_ids

            success, matched_tokens = self.cache_manager.allocate_request(
                req.request_id,
                req.total_length,
                token_ids=token_ids,
            )
            if not success:
                raise RuntimeError(f"OOM during prefill allocation for {req.request_id}")

            prefix_lens_list.append(matched_tokens)

            if self.is_first_peer:
                if matched_tokens > 0 and self.enable_prefix_cache:
                    # Skip the prefix tokens that are already cached
                    # But we must keep at least the last token to generate next token logits
                    new_tokens = req.input_ids[matched_tokens:]
                    if len(new_tokens) == 0:
                        # All tokens cached - keep the last token and adjust prefix_len
                        new_tokens = req.input_ids[-1:]
                        prefix_lens_list[-1] = matched_tokens - 1
                        actual_processed_lengths_list.append(1)
                        logger.debug(
                            f"Request {req.request_id}: Full cache hit, keeping last token for logits"
                        )
                    else:
                        actual_processed_lengths_list.append(len(new_tokens))
                        logger.debug(
                            f"Request {req.request_id}: Skipping {matched_tokens} cached tokens, "
                            f"processing {len(new_tokens)} new tokens"
                        )
                    h_or_tokens_list.append(new_tokens)
                else:
                    h_or_tokens_list.append(req.input_ids)
                    actual_processed_lengths_list.append(len(req.input_ids))
            else:
                if matched_tokens > 0 and self.enable_prefix_cache:
                    keep_len = req.total_length - matched_tokens
                    if keep_len <= 0:
                        # All tokens cached - keep the last hidden state
                        new_hidden = req.hidden_states[-1:]
                        prefix_lens_list[-1] = matched_tokens - 1
                        actual_processed_lengths_list.append(1)
                    else:
                        new_hidden = req.hidden_states[-keep_len:]
                        actual_processed_lengths_list.append(new_hidden.shape[0])
                    h_or_tokens_list.append(new_hidden)
                else:
                    h_or_tokens_list.append(req.hidden_states)
                    actual_processed_lengths_list.append(req.hidden_states.shape[0])

            block_table = self.cache_manager.get_block_table(req.request_id)
            block_tables_list.append(block_table)
            # For prefill, context length after this step will be total_length
            context_lengths_list.append(req.total_length)

        if self.is_first_peer:
            padded_inputs, padding_mask = pad_inputs(
                self.pad_token_id, h_or_tokens_list, self.dtype
            )
        else:
            padded_inputs, padding_mask = pad_inputs(0, h_or_tokens_list, self.dtype)

        slot_mapping_tensor = None
        if self.cache_manager.needs_blocks:
            # Generate slot_mapping for prefill (only for NEW tokens, starting from prefix_len)
            max_len = padded_inputs.shape[1]
            slot_mapping_flat = []

            for i, req in enumerate(batched_requests):
                block_table = block_tables_list[i]
                prefix_len = prefix_lens_list[i]
                total_len = req.total_length
                new_tokens_len = total_len - prefix_len

                for seq_idx in range(max_len):
                    if seq_idx < new_tokens_len:
                        # Valid new token - map to position after prefix
                        actual_pos = prefix_len + seq_idx
                        block_idx = actual_pos // self.cache_manager.block_size
                        block_offset = actual_pos % self.cache_manager.block_size
                        physical_block = block_table[block_idx]
                        slot = physical_block * self.cache_manager.block_size + block_offset
                        slot_mapping_flat.append(slot)
                    else:
                        # Padding token. The kernel should ignore this.
                        slot_mapping_flat.append(-1)

            slot_mapping_tensor = mx.array(slot_mapping_flat, dtype=mx.int64)

        # Pad block tables. Linear-only shards do not allocate KV blocks.
        max_blocks = max(len(bt) for bt in block_tables_list)
        padded_block_tables = []
        for bt in block_tables_list:
            padded_block_tables.append(bt + [0] * (max_blocks - len(bt)))

        block_tables_tensor = mx.array(padded_block_tables, dtype=mx.int32)
        context_lengths_tensor = mx.array(context_lengths_list, dtype=mx.int32)

        # Create mask for standard attention (used during Prefill computation)
        causal_mask = create_causal_mask(padded_inputs.shape[1], padded_inputs.shape[1], self.dtype)
        mask = combine_padding_and_causal_masks(padding_mask, causal_mask, self.dtype)

        # Prepare state slot mapping if needed
        state_slot_mapping = None
        if self.cache_manager.needs_slots:
            req_ids = [r.request_id for r in batched_requests]
            slots = [self.cache_manager.get_slot(rid) for rid in req_ids]
            state_slot_mapping = mx.array(slots, dtype=mx.int32)

        # Convert prefix_lens to tensor for models that need RoPE offset adjustment
        prefix_lens_tensor = mx.array(prefix_lens_list, dtype=mx.int32)
        # Convert actual_processed_lengths to tensor for correct logit selection
        actual_processed_lengths_tensor = mx.array(actual_processed_lengths_list, dtype=mx.int32)

        ret = {
            "h_or_tokens": padded_inputs,
            "cache": self.cache_manager.get_caches(),
            "mask": mask,
            "requests": batched_requests,
            "block_tables": block_tables_tensor,
            "context_lengths": context_lengths_tensor,
            "slot_mapping": slot_mapping_tensor,
            "state_slot_mapping": state_slot_mapping,
            "prefix_lens": prefix_lens_tensor,  # For RoPE offset calculation
            "actual_processed_lengths": actual_processed_lengths_tensor,  # For correct logit selection
        }
        logger.debug(f"Prepared MLX prefill batch (size={batch_size})")
        return ret

    def _prepare_inline_prefill_batch(self, batched_requests: List[Request]) -> Optional[Dict[str, Any]]:
        """A prefill batch whose prompts are run through the shard in slices; see parallax.server.inline_prefill."""
        if not batched_requests:
            return None
        for req in batched_requests:
            assert req.is_prefill, f"Request {req.request_id} is not a prefill request."
            success, _ = self.cache_manager.allocate_request(req.request_id, req.total_length)
            if not success:
                raise RuntimeError(f"OOM during prefill allocation for {req.request_id}")
        lengths = [req.total_length for req in batched_requests]
        return {
            "requests": batched_requests,
            "inline_prompts": [list(req.input_ids) for req in batched_requests],
            "context_lengths": mx.array(lengths, dtype=mx.int32),
            "prefix_lens": mx.zeros((len(lengths),), dtype=mx.int32),
            "actual_processed_lengths": mx.array(lengths, dtype=mx.int32),
        }

    def _prefill_inline(self, prepared_inputs: Dict[str, Any]) -> mx.array:
        """The rows of every request's prompt, run in slices, padded on the right to the longest."""
        rows = [
            prefill_in_slices(
                self.model_shard,
                self.cache_manager,
                req.request_id,
                prompt,
                self.inline_prefill_chunk_size,
                self.dtype,
                self.pad_token_id,
            )
            for req, prompt in zip(prepared_inputs["requests"], prepared_inputs["inline_prompts"], strict=True)
        ]
        width = max(row.shape[1] for row in rows)
        return mx.concatenate(
            [mx.pad(row, [(0, 0), (0, width - row.shape[1]), (0, 0)]) for row in rows], axis=0
        )

    def _is_verify_request(self, request: Request) -> bool:
        return self.speculation is not None and self.speculation.is_speculating(request.request_id)

    def _prepare_verify_batch(self, batched_requests: List[Request]) -> Optional[Dict[str, Any]]:
        """Inputs for the speculative steps of requests whose last peer proposed drafts.

        Each request runs its pending tokens, its input token and its drafts as one chunk that continues from the tokens
        this peer has absorbed, the way a later chunk of a chunked prefill does.
        """
        if not batched_requests:
            return None
        steps: List[Step] = []
        for req in batched_requests:
            steps.append(self.speculation.begin(req.request_id, req.output_ids[-1]))
            self._verify_steps[req.request_id] = steps[-1]
        inputs = continuation_inputs(
            self.cache_manager,
            [req.request_id for req in batched_requests],
            [step.tokens for step in steps],
            [step.prefix_len for step in steps],
            self.dtype,
            self.pad_token_id,
        )
        return {**inputs, "requests": batched_requests, "verify_steps": steps}

    def _prepare_decode_batch(self, batched_requests: List[Request]) -> Optional[Dict[str, Any]]:
        """Prepares inputs for ShardedModel from a batch of decode requests."""
        batch_size = len(batched_requests)
        if batch_size == 0:
            return None

        h_or_tokens_list = []
        block_tables_list = []
        context_lengths_list = []
        valid_requests = []

        for req in batched_requests:
            assert req.is_decoding, f"Request {req.request_id} is not a decode request."

            # Allocate slot for new token
            success = self.cache_manager.append_slot(req.request_id)
            if not success:
                logger.error(
                    f"OOM during decode for {req.request_id}. Aborting request and notifying other nodes."
                )
                req.update_status(RequestStatus.FINISHED_ABORT)
                self.cache_manager.free_request(req.request_id)
                self.scheduler.evict_request(req.request_id)
                # Add to finished_batch to trigger abort notification
                self.finished_batch.append(req)

                # If this is First Peer, notify the Rust frontend immediately.
                if self.is_first_peer and self.tp_rank == 0:
                    self.send_engine_core_request_output(
                        request=req,
                        token_id=None,
                    )

                continue

            # Allocation successful, proceed with batch preparation
            valid_requests.append(req)

            if self.is_first_peer:
                # First peer input is the last generated token
                h_or_tokens_list.append([req.output_ids[-1]])
            else:
                h_or_tokens_list.append(req.hidden_states)

            # Update token_ids for prefix cache (if enabled)
            if self.enable_prefix_cache and self.is_first_peer:
                # Add the new token to the request's token_ids
                self.cache_manager.update_request_tokens(req.request_id, [req.output_ids[-1]])

            block_table = self.cache_manager.get_block_table(req.request_id)
            block_tables_list.append(block_table)
            context_lengths_list.append(self.cache_manager.get_context_length(req.request_id))

        # Check if we have any valid requests left
        if not valid_requests:
            return None

        batch_size = len(valid_requests)

        if isinstance(h_or_tokens_list[0], list):
            # First peer case: h_or_tokens_list is list of list of ints [[token_id], ...]
            padded_inputs = mx.array(h_or_tokens_list, dtype=mx.int32)  # (Batch, 1)
        else:
            # Intermediate peer case: h_or_tokens_ list is list of mx.arrays (1, D)
            padded_inputs = mx.concatenate(h_or_tokens_list, axis=0)  # (Batch, D)
            padded_inputs = padded_inputs.reshape(batch_size, 1, -1)  # (Batch, 1, D)

        # Pad block tables. Linear-only shards do not allocate KV blocks.
        max_blocks = max(len(bt) for bt in block_tables_list)
        padded_block_tables = []
        for bt in block_tables_list:
            padded_block_tables.append(bt + [0] * (max_blocks - len(bt)))

        block_tables_tensor = mx.array(padded_block_tables, dtype=mx.int32)
        context_lengths_tensor = mx.array(context_lengths_list, dtype=mx.int32)

        slot_mapping_tensor = None
        if self.cache_manager.needs_blocks:
            slots = []
            for block_table, context_len in zip(block_tables_list, context_lengths_list):
                token_pos = context_len - 1
                block_idx = token_pos // self.cache_manager.block_size
                block_offset = token_pos % self.cache_manager.block_size
                slots.append(block_table[block_idx] * self.cache_manager.block_size + block_offset)
            slot_mapping_tensor = mx.array(slots, dtype=mx.int64)

        # Prepare state slot mapping if needed
        state_slot_mapping = None
        if self.cache_manager.needs_slots:
            req_ids = [r.request_id for r in valid_requests]
            slots = [self.cache_manager.get_slot(rid) for rid in req_ids]
            state_slot_mapping = mx.array(slots, dtype=mx.int32)

        ret = {
            "h_or_tokens": padded_inputs,
            "cache": self.cache_manager.get_caches(),
            "mask": None,
            "requests": valid_requests,
            "block_tables": block_tables_tensor,
            "context_lengths": context_lengths_tensor,
            "slot_mapping": slot_mapping_tensor,
            "state_slot_mapping": state_slot_mapping,
        }
        logger.debug(f"Prepared MLX decode batch (size={batch_size})")
        return ret
