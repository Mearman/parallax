from __future__ import annotations

import json
import os
import shutil
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.distributed
import vllm.distributed.parallel_state as parallel_state
from mlx_lm.utils import load_config
from vllm.config import (
    CacheConfig,
    DeviceConfig,
    LoadConfig,
    LoRAConfig,
    ModelConfig,
    ParallelConfig,
    SchedulerConfig,
    SpeculativeConfig,
    VllmConfig,
    set_current_vllm_config,
)
from vllm.distributed.kv_transfer.kv_connector.utils import get_current_attn_backends
from vllm.distributed.parallel_state import GroupCoordinator as VLLMGroupCoordinator
from vllm.lora.request import LoRARequest
from vllm.platforms import current_platform
from vllm.v1.attention.backends.utils import (
    get_supported_kv_cache_layouts,
    record_kv_cache_layout,
    resolve_kv_cache_layout,
)
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.core.kv_cache_utils import (
    generate_scheduler_kv_cache_config,
    get_kv_cache_configs,
)
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.worker.gpu_model_runner import GPUModelRunner
from vllm.v1.worker.workspace import current_workspace_manager, init_workspace_manager

from parallax.sglang.monkey_patch_utils.weight_loader_filter import (
    apply_weight_loader_filter_patch,
    set_layer_range_for_filtering,
)
from parallax.utils.tokenizer_utils import load_tokenizer
from parallax.utils.utils import normalize_model_config
from parallax.vllm.monkey_patch import apply_parallax_vllm_monkey_patch
from parallax_utils.logging_config import get_logger
from parallax_utils.prepare_adapter import download_adapter_config

logger = get_logger(__name__)


def resolve_cache_layout(vllm_config, kv_cache_specs) -> None:
    """Resolve the KV cache layout once, as vLLM's engine core does before it sizes the cache.

    This runner is the only worker, so the backends it reports are the whole agreement the core would
    collect. A layout already on the cache config is kept, which makes a second call a no-op.
    """
    backends = get_current_attn_backends(vllm_config)
    supported = [[layout.name for layout in get_supported_kv_cache_layouts(backends)]]
    layout = resolve_kv_cache_layout(vllm_config, supported, kv_cache_specs.values())
    record_kv_cache_layout(vllm_config.cache_config, layout.name)


def adopt_kv_cache_config(vllm_config, kv_cache_config) -> None:
    """Copy onto the cache config what vLLM's worker records when it initialises from a KV cache config.

    The block count, hash block size and cache-hit alignment are read from the cache config later,
    during warmup and by the cache manager.
    """
    cache_config = vllm_config.cache_config
    cache_config.num_gpu_blocks = kv_cache_config.num_blocks
    cache_config.hash_block_size = kv_cache_config.hash_block_size
    cache_config.cache_hit_alignment_tokens = kv_cache_config.cache_hit_alignment_tokens


class ParallaxVLLMGroupCoordinator(VLLMGroupCoordinator):
    """
    Parallax version of vLLM's GroupCoordinator.
    Override is_first_rank and is_last_rank to use layer ranges instead of process ranks.
    """

    def __init__(
        self,
        group_ranks: List[List[int]],
        local_rank: int,
        torch_distributed_backend: Union[str, torch.distributed.Backend],
        use_device_communicator: bool,
        use_message_queue_broadcaster: bool = False,
        group_name: Optional[str] = None,
        pp_start_layer: int = 0,
        pp_end_layer: int = 0,
        num_hidden_layers: int = 0,
    ):
        super().__init__(
            group_ranks=group_ranks,
            local_rank=local_rank,
            torch_distributed_backend=torch_distributed_backend,
            use_device_communicator=use_device_communicator,
            use_message_queue_broadcaster=use_message_queue_broadcaster,
            group_name=group_name,
        )
        self.pp_start_layer = pp_start_layer
        self.pp_end_layer = pp_end_layer
        self.num_hidden_layers = num_hidden_layers

    @property
    def is_first_rank(self) -> bool:
        """Return whether this is the first pipeline stage based on layer range."""
        return self.pp_start_layer == 0

    @property
    def is_last_rank(self) -> bool:
        """Return whether this is the last pipeline stage based on layer range."""
        return self.pp_end_layer >= self.num_hidden_layers


class ParallaxVLLMModelRunner(GPUModelRunner):

    def __init__(
        self,
        vllm_config: VllmConfig,
        kv_cache_config: Optional[KVCacheConfig],
        device: torch.device,
        start_layer: int,
        end_layer: int,
        num_hidden_layers: int,
    ):
        self.start_layer = start_layer
        self.end_layer = end_layer
        self.num_hidden_layers = num_hidden_layers
        self.num_shard_layers = end_layer - start_layer

        self.is_first_peer = start_layer == 0
        self.is_last_peer = end_layer == num_hidden_layers

        self.pp_rank = 0
        self.pp_size = 1

        self.request_block_hasher: Optional[Callable[[Any], List[Any]]] = None
        self.enable_prefix_caching: bool = False
        self.lora_history: List[Tuple[str, int, str]] = []  # lora_name, lora_id, lora_path
        # Draft tokens the speculative drafter proposed for each request's next step; the decode batch schedules them
        # and the target model verifies them.
        self.pending_drafts: Dict[str, List[int]] = {}

        super().__init__(vllm_config=vllm_config, device=device)
        self.kv_cache_config = kv_cache_config

        logger.info(
            f"ParallaxVLLMModelRunner initialized: layers [{start_layer}, {end_layer}), "
            f"is_first={self.is_first_peer}, is_last={self.is_last_peer}"
        )

    def initialize_kv_cache_manager(self, max_model_len: int) -> KVCacheManager:
        logger.debug("Initializing vLLM KVCacheManager...")

        if self.kv_cache_config is None:
            raise RuntimeError("the KV cache config must be generated before the KV cache manager is built")

        # vLLM's scheduler falls back to the block size when the cache config carries no hash block size.
        hash_block_size = self.kv_cache_config.hash_block_size
        if hash_block_size is None:
            hash_block_size = self.vllm_config.cache_config.block_size

        kv_cache_manager = KVCacheManager(
            kv_cache_config=self.kv_cache_config,
            max_model_len=max_model_len,
            enable_caching=self.enable_prefix_caching,
            use_eagle=False,
            log_stats=True,
            enable_kv_cache_events=False,
            dcp_world_size=1,
            scheduler_block_size=self.vllm_config.cache_config.block_size,
            hash_block_size=hash_block_size,
        )

        self.kv_cache_manager = kv_cache_manager

        return kv_cache_manager

    def load_model(self, load_dummy_weights: bool = False, *, model=None) -> None:
        logger.debug(f"Loading vLLM model with layers [{self.start_layer}, {self.end_layer})...")

        from vllm.distributed.utils import get_pp_indices

        original_get_pp_indices = get_pp_indices

        def custom_get_pp_indices(num_layers: int, rank: int, world_size: int):
            logger.debug(
                f"custom_get_pp_indices called: num_layers={num_layers}, "
                f"returning [{self.start_layer}, {self.end_layer})"
            )
            return self.start_layer, self.end_layer

        import vllm.distributed.utils

        vllm.distributed.utils.get_pp_indices = custom_get_pp_indices

        try:
            super().load_model(load_dummy_weights, model=model)
            logger.debug(
                f"Successfully loaded {self.num_shard_layers} layers "
                f"[{self.start_layer}:{self.end_layer}]"
            )

        finally:
            vllm.distributed.utils.get_pp_indices = original_get_pp_indices

    def execute_model(
        self, scheduler_output, intermediate_tensors=None, return_decoded_tokens=False
    ):
        """
        Execute the model with the given scheduler output and intermediate tensors.
        If this is not the first peer, and the intermediate_tensors buffer is not initialized,
        initialize it.
        """
        if not self.is_first_peer and self.intermediate_tensors is None:
            self.intermediate_tensors = self.model.make_empty_intermediate_tensors(
                batch_size=self.max_num_tokens,
                dtype=self.model_config.dtype,
                device=self.device,
            )
            logger.debug("Successfully initialized intermediate_tensors buffer")

        super().execute_model(scheduler_output, intermediate_tensors)

        sampled_token_ids = None
        sampler_output = None
        logits = None

        if return_decoded_tokens:
            if hasattr(self.execute_model_state, "logits"):
                logits = self.execute_model_state.logits

            sampler_output = super().sample_tokens(grammar_output=None)
            sampled_token_ids = sampler_output._sampled_token_ids
            sampled_token_ids_cpu = sampler_output.sampled_token_ids_cpu
            drafts = self.take_draft_token_ids()
            if drafts is not None:
                for request_id, draft_token_ids in zip(drafts.req_ids, drafts.draft_token_ids):
                    self.pending_drafts[request_id] = [int(token) for token in draft_token_ids]

        return (
            self.execute_model_state,
            sampled_token_ids,
            sampled_token_ids_cpu,
            sampler_output,
            logits,
        )


def _init_and_reserve_workspace(device: torch.device, max_num_tokens: int) -> None:

    init_workspace_manager(device)

    try:
        _MB = 1024**2
        per_token_workspace = 24 * 1024
        estimated_workspace = max_num_tokens * per_token_workspace
        reserve_size = max(512 * _MB, min(estimated_workspace, 8192 * _MB))
        free_mem, _ = torch.cuda.mem_get_info(device.index)
        if reserve_size > free_mem * 0.5:
            logger.warning(
                f"Estimated workspace ({reserve_size / _MB:.0f}MB) is >50% of free memory "
                f"({free_mem / _MB:.0f}MB). Clamping to 4GB to preserve memory for KV cache."
            )
            reserve_size = min(reserve_size, 4096 * _MB)

        current_workspace_manager()._ensure_workspace_size(reserve_size)
        logger.info(
            f"Initialized WorkspaceManager and reserved {reserve_size / _MB:.2f} MB buffer "
            f"(max_num_tokens={max_num_tokens})"
        )
    except Exception as e:
        logger.warning(f"Failed to reserve workspace buffer: {e}")


def initialize_vllm_model_runner(
    model_repo: str,
    start_layer: int,
    end_layer: int,
    kv_cache_memory_fraction: float,
    attention_backend: str,
    max_batch_size: int,
    max_sequence_length: int,
    max_num_tokens_per_batch: int = 16384,
    dtype: str = "float16",
    moe_runner_backend: str = "auto",
    tp_rank: int = 0,
    tp_size: int = 1,
    nccl_port: int = None,
    enable_return_routed_experts: bool = False,
    instance_id: Optional[str] = None,
    **kwargs,
) -> Tuple[ParallaxVLLMModelRunner, Dict, Any]:
    from parallax.utils.model_download import selective_model_download

    logger.info(
        f"Initializing vLLM model runner for {model_repo}, " f"layers=[{start_layer}, {end_layer})"
    )

    model_path = selective_model_download(
        model_repo,
        start_layer=start_layer,
        end_layer=end_layer,
    )

    # ParallaxVLLMModelRunner subclasses vLLM's v1 GPUModelRunner. The v2 runner, which newer vLLM selects by default,
    # prepares per-step state (the hybrid models' aligned state indices) that only it provides, so the config must
    # say v1 before it is built.
    os.environ["VLLM_USE_V2_MODEL_RUNNER"] = "0"

    # Multimodal-style checkpoints nest the text model's fields; the layer count is read from the flattened config, as the SGLang runner does.
    config = normalize_model_config(load_config(model_path))
    tokenizer = load_tokenizer(model_path, eos_token_ids=config.get("eos_token_id", None))
    dtype = config.get("torch_dtype", "bfloat16")

    num_hidden_layers = config.get("num_hidden_layers")
    is_first_peer = start_layer == 0
    is_last_peer = end_layer == num_hidden_layers

    # Apply Parallax vLLM monkey patches for pipeline parallelism
    try:
        apply_parallax_vllm_monkey_patch(is_first_stage=is_first_peer, is_last_stage=is_last_peer)
        logger.debug(
            f"Applied Parallax vLLM monkey patches: is_first_stage={is_first_peer}, is_last_stage={is_last_peer}"
        )
    except Exception as e:
        logger.warning("Failed to apply Parallax vLLM monkey patches: %s", e)

    # Apply layer-range-based weight file filtering before any model load.
    # Reuse the generic monkey patch used by sglang implementation to reduce
    # local weight file reads when loading a partial layer shard.
    try:
        set_layer_range_for_filtering(start_layer, end_layer, num_hidden_layers)
        apply_weight_loader_filter_patch()
        logger.debug(
            f"Applied weight loader filter monkey patch for layers [{start_layer}, {end_layer})"
        )
    except Exception as e:
        logger.warning("Failed to apply weight loader filter patch for vLLM loading: %s", e)

    # For single process, always use pp_size=1
    virtual_pp_size = 1

    # Set TP device
    device = torch.device(f"cuda:{tp_rank}")
    torch.cuda.set_device(device)

    if enable_return_routed_experts:
        raise NotImplementedError("routed-experts capture is not supported by this vLLM's ModelConfig")

    if end_layer > num_hidden_layers:
        raise ValueError(
            f"end_layer ({end_layer}) cannot be greater than "
            f"num_hidden_layers ({num_hidden_layers})"
        )

    if max_sequence_length is not None:
        max_len = max_sequence_length
    else:
        max_len = getattr(config, "max_position_embeddings", 4096)

    model_config = ModelConfig(
        model=str(model_path),
        tokenizer=str(model_path),
        tokenizer_mode="auto",
        trust_remote_code=True,
        dtype=dtype,
        seed=0,
        max_model_len=max_len,
        max_logprobs=1,
    )

    cache_config = CacheConfig(
        # Left unset: in the align cache mode vLLM enlarges the attention block of a hybrid (mamba) model until its
        # page matches the recurrent state's page. A fixed small block is padded to that page size instead.
        gpu_memory_utilization=kv_cache_memory_fraction,
        cache_dtype="auto",
    )

    parallel_config = ParallelConfig(
        pipeline_parallel_size=virtual_pp_size,
        tensor_parallel_size=tp_size,
        node_rank=tp_rank,
        rank=tp_rank,
        distributed_executor_backend=None,
    )

    device_config = DeviceConfig(device=device)
    load_config_for_config = LoadConfig(load_format="auto")

    max_batched_tokens = max(max_num_tokens_per_batch, model_config.max_model_len)
    max_num_seqs = max_batch_size

    scheduler_config = SchedulerConfig(
        max_num_batched_tokens=max_batched_tokens,
        max_num_seqs=max_num_seqs,
        max_model_len=model_config.max_model_len,
        is_encoder_decoder=False,
        # The align cache mode that hybrid (mamba) models use with vLLM's default prefix caching requires it.
        enable_chunked_prefill=True,
    )

    # LoRA Config construction
    enable_lora = kwargs.get("enable_lora", False)
    lora_config = None
    lora_req = None
    lora_name = None
    lora_int_id = None
    lora_path = None
    if enable_lora:
        # Hard code a large moe chunk size. Need to improve this.
        if "VLLM_FUSED_MOE_CHUNK_SIZE" not in os.environ:
            os.environ["VLLM_FUSED_MOE_CHUNK_SIZE"] = "65536"

        max_lora_rank = kwargs.get("max_lora_rank")
        if max_lora_rank is None:
            max_lora_rank = 64
            logger.warning(f"max_lora_rank not specified, using default: {max_lora_rank}")

        max_loras = kwargs.get("max_loras_per_batch", 1)
        if max_loras is None:
            max_loras = 1

        max_cpu_loras = kwargs.get("max_loaded_loras")
        fully_sharded_loras = kwargs.get("fully_sharded_loras", False)

        lora_path = kwargs.get("lora_path")

        lora_config = LoRAConfig(
            max_lora_rank=max_lora_rank,
            max_loras=max_loras,
            fully_sharded_loras=fully_sharded_loras,
            max_cpu_loras=max_cpu_loras,
            lora_dtype=dtype,
        )
        logger.info(f"LoRA config: {lora_config}")

        # Create a simple hash or ID for the LoRA based on path
        # In a real scenario, we might want a more robust ID mapping mechanism
        lora_name = f"lora_{hash(lora_path) % 10000}"
        lora_int_id = abs(hash(lora_path)) % 10000 + 1

        lora_req = LoRARequest(lora_name=lora_name, lora_int_id=lora_int_id, lora_path=lora_path)
        logger.debug(f"Created LoRA request: {lora_name} (id={lora_int_id}) path={lora_path}")

        # Workaround: save adapter_config.json locally for lora update
        download_adapter_config(lora_path)

    speculative_config = None
    speculative_json = kwargs.get("speculative_config")
    if speculative_json:
        speculative = json.loads(speculative_json)
        speculative.update(
            {"target_model_config": model_config, "target_parallel_config": parallel_config}
        )
        speculative_config = SpeculativeConfig(**speculative)

    vllm_config = VllmConfig(
        model_config=model_config,
        cache_config=cache_config,
        parallel_config=parallel_config,
        scheduler_config=scheduler_config,
        device_config=device_config,
        load_config=load_config_for_config,
        lora_config=lora_config,
        speculative_config=speculative_config,
        quant_config=None,
        kv_transfer_config=None,
        kv_events_config=None,
        additional_config={},
        instance_id=instance_id or "",
    )

    # vLLM's parallel-state setup reads the current config, so it runs inside the config context.
    with set_current_vllm_config(vllm_config):
        if not parallel_state.model_parallel_is_initialized():
            logger.debug(f"Initializing vLLM distributed environment...")

            # Set environment variables for distributed initialization
            if "RANK" not in os.environ:
                os.environ["RANK"] = str(tp_rank)
            if "WORLD_SIZE" not in os.environ:
                os.environ["WORLD_SIZE"] = str(tp_size)
            if "LOCAL_RANK" not in os.environ:
                os.environ["LOCAL_RANK"] = str(tp_rank)
            if "MASTER_ADDR" not in os.environ:
                os.environ["MASTER_ADDR"] = "localhost"
            if "MASTER_PORT" not in os.environ:
                os.environ["MASTER_PORT"] = str(nccl_port)

            try:
                parallel_state.init_distributed_environment()

                # Initialize with pp_size=1 for single process
                parallel_state.initialize_model_parallel(
                    tensor_model_parallel_size=1,
                    pipeline_model_parallel_size=1,
                )

                # Monkey patch the PP group with our custom Parallax coordinator
                # that uses layer ranges to determine is_first_rank/is_last_rank
                original_pp_group = parallel_state._PP
                if original_pp_group is not None:
                    # Get backend from device_group (torch is already imported at module level)

                    backend = torch.distributed.get_backend(original_pp_group.device_group)

                    # Create a Parallax PP group coordinator
                    # Need to wrap ranks in a list of lists for group_ranks parameter
                    parallax_pp_group = ParallaxVLLMGroupCoordinator(
                        group_ranks=[original_pp_group.ranks],
                        local_rank=original_pp_group.local_rank,
                        torch_distributed_backend=backend,
                        use_device_communicator=original_pp_group.use_device_communicator,
                        use_message_queue_broadcaster=(original_pp_group.mq_broadcaster is not None),
                        group_name="pp",
                        pp_start_layer=start_layer,
                        pp_end_layer=end_layer,
                        num_hidden_layers=num_hidden_layers,
                    )
                    # Replace the PP group
                    parallel_state._PP = parallax_pp_group
                    logger.debug(
                        f"Replaced vLLM PP group with Parallax coordinator: "
                        f"is_first_rank={parallax_pp_group.is_first_rank}, "
                        f"is_last_rank={parallax_pp_group.is_last_rank}"
                    )

                logger.debug(f"vLLM distributed environment initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize distributed environment: {e}")
                logger.error(f"vLLM distributed initialization failed. Error: {e}")
                raise


    model_runner = ParallaxVLLMModelRunner(
        vllm_config=vllm_config,
        kv_cache_config=None,
        device=device,
        start_layer=start_layer,
        end_layer=end_layer,
        num_hidden_layers=num_hidden_layers,
    )

    with set_current_vllm_config(vllm_config):
        logger.info("Loading vLLM model (partial layers)...")
        model_runner.load_model()
        logger.info("vLLM model loaded successfully")

        # vLLM's executors do this right after the model loads: it settles the attention block size, and for a
        # hybrid (mamba) model aligns it so the attention page equals the recurrent state's page.
        current_platform.update_block_size_for_backend(vllm_config)

        logger.debug("Letting vLLM automatically generate KV cache configuration...")

        # Init workspace manager for capturing graph and reserve memory
        # We do this BEFORE calculating available memory for KV cache to avoid OOM
        _init_and_reserve_workspace(device, max_num_tokens_per_batch)

        kv_cache_specs = model_runner.get_kv_cache_spec()

        if not kv_cache_specs:
            raise RuntimeError("No KV cache specs found in the loaded model")

        free_memory, _ = torch.cuda.mem_get_info(device.index)
        available_memory = int(free_memory * kv_cache_memory_fraction)

        logger.info(
            f"Available GPU memory for KV cache: "
            f"{available_memory / (1024**3):.2f} GB "
            f"({kv_cache_memory_fraction:.1%} of {free_memory / (1024**3):.2f} GB)"
        )

        resolve_cache_layout(model_runner.vllm_config, kv_cache_specs)
        kv_cache_configs = get_kv_cache_configs(
            vllm_config=model_runner.vllm_config,
            kv_cache_specs=[kv_cache_specs],
            available_memory=[available_memory],
        )

        kv_cache_config = generate_scheduler_kv_cache_config(kv_cache_configs)
        adopt_kv_cache_config(model_runner.vllm_config, kv_cache_config)
        logger.info(
            f"KV cache allocated: {kv_cache_config.num_blocks} blocks of "
            f"{model_runner.vllm_config.cache_config.block_size} tokens"
        )

        model_runner.kv_cache_config = kv_cache_config

        logger.info("Initializing GPUModelRunner KV cache...")
        model_runner.initialize_kv_cache(kv_cache_config)
        logger.info("GPUModelRunner KV cache initialized successfully")

        logger.info("Initializing KV Cache Manager...")
        model_runner.initialize_kv_cache_manager(max_model_len=model_config.max_model_len)
        logger.info("KV Cache Manager initialized successfully")

        # Warm up the model and capture CUDA graphs if enabled
        # This prevents the first request from triggering compilation/graph capture
        logger.info("Warming up model and capturing CUDA graphs...")
        try:
            # Create a dedicated stream for graph capture to avoid "non-default stream" error
            with torch.cuda.stream(torch.cuda.Stream(device=device)):
                model_runner.capture_model()
            torch.cuda.current_stream(device).synchronize()
            logger.info("Model warmup and CUDA graph capture completed successfully")
        except Exception as e:
            logger.warning(f"Failed to capture CUDA graph during initialization: {e}")

        if enable_lora:
            logger.info(f"Initializing lora adapters...")
            model_runner.add_lora(lora_req)
            model_runner.default_lora_req = lora_req

            lora_info = (lora_name, lora_int_id, lora_path)
            model_runner.lora_history.append(lora_info)

    return model_runner, config, tokenizer


def refit_vllm_model(
    model_runner: ParallaxVLLMModelRunner,
    tensors: dict = None,
    refit_weight_path: str = None,
):
    """Runtime weight refit from disk"""
    if tensors is not None:
        logger.info(f"Executor begins weight refit from host memory")
        for x in tensors.keys():
            refit_tensors = [(x, tensors.get(x))]
            model_runner.model.load_weights(weights=refit_tensors)
    elif refit_weight_path is not None:
        logger.info(f"Executor begins weight refit from disk files")
        # config_overrides = {"load_config": {"download_dir": refit_weight_path}}
        # model_runner.update_config(overrides=config_overrides)
        # model_runner.reload_weights()
        adapter_path = os.path.join(os.getcwd(), "adapter_config.json")
        if os.path.isfile(adapter_path):
            shutil.copy(adapter_path, refit_weight_path)
        else:
            logger.warning(f"Cannot find adapter_config.json locally. Exit lora weight refit.")
            return

        lora_name = f"lora_{hash(refit_weight_path) % 10000}"
        lora_int_id = abs(hash(refit_weight_path)) % 10000 + 1
        lora_req = LoRARequest(
            lora_name=lora_name, lora_int_id=lora_int_id, lora_path=refit_weight_path
        )
        logger.info(
            f"Created LoRA request: {lora_name} (id={lora_int_id}) path={refit_weight_path}"
        )

        # Release old loras if needed
        before_loras = model_runner.list_loras()
        history = model_runner.lora_history
        assert len(before_loras) == len(
            history
        ), f"Before lora refit, number of loaded lora mismatch!"
        logger.info(f"Before lora refit number of lora adapters: {len(before_loras)}")
        while len(history) > 1:
            _, old_lora_id, _ = history.pop(0)
            model_runner.remove_lora(old_lora_id)

        # Add new lora
        model_runner.add_lora(lora_req)
        model_runner.default_lora_req = lora_req
        lora_info = (lora_name, lora_int_id, refit_weight_path)
        history.append(lora_info)
        model_runner.lora_history = history

        # Check lora slots
        after_loras = model_runner.list_loras()
        after_history = model_runner.lora_history
        assert len(after_loras) == len(
            after_history
        ), f"After lora refit, number of loaded lora mismatch!"
        logger.info(f"After lora refit number of lora adapters: {len(after_loras)}")
    else:
        assert False, "Weight refit needs host tensors or weight path"
    logger.info(f"Finish weight refit")
