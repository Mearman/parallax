"""
Imports sglang ModelRunner related modules and wrap them into create functions.
We use monkey patch to modify sglang originated methods. The main purpose is to pass
arguments needed by decentralized inference.
"""

import logging
import random
from typing import List, Optional

import torch
from mlx_lm.utils import load_config
from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.distributed.bootstrap import init_layer_runtime, init_parallel_runtime
from sglang.srt.layers.moe import initialize_moe_config
from sglang.srt.model_executor.model_runner import ModelRunner as SGLModelRunner
from sglang.srt.model_executor.runner_utils.pool import prewarm_graph_pool_borrow
from sglang.srt.runtime_context import SpawnRanks, get_device, get_parallel, publish
from sglang.srt.server_args import ServerArgs

from parallax.sglang.monkey_patch import apply_parallax_sglang_monkey_patch
from parallax.sglang.monkey_patch_utils.model_parallel import (
    VIRTUAL_PIPELINE_WIDTH,
    install_layer_range,
)
from parallax.sglang.monkey_patch_utils.weight_loader_filter import (
    set_layer_range_for_filtering,
)
from parallax.utils.tokenizer_utils import load_tokenizer
from parallax.utils.utils import normalize_model_config

logger = logging.getLogger(__name__)



class ParallaxModelRunner(SGLModelRunner):
    """Parallax ModelRunner module.

    The runner owns one contiguous layer range of the model. SGLang builds its process groups
    from the published runtime configuration, so this publishes the configuration, brings the
    parallel runtime up exactly as the scheduler entry does, and then records the layer range on
    the pipeline group, which is where the model code and the patched layer builder read it.
    """

    def __init__(
        self,
        model_config: ModelConfig,
        mem_fraction_static: float,
        gpu_id: int,
        nccl_port: int,
        server_args: ServerArgs,
        pp_start_layer: int,
        pp_end_layer: int,
        tp_rank: int = 0,
    ):
        """Publish the runtime, build the groups for this layer range, then load the model."""
        self.pp_start_layer = pp_start_layer
        self.pp_end_layer = pp_end_layer
        num_hidden_layers = model_config.hf_config.num_hidden_layers
        set_layer_range_for_filtering(pp_start_layer, pp_end_layer, num_hidden_layers)

        publish(
            server_args,
            role="scheduler",
            ranks=SpawnRanks(world_rank=tp_rank, gpu_id=gpu_id),
        )
        init_parallel_runtime(
            server_args=server_args,
            device=get_device().device,
            dist_port=nccl_port,
        )
        init_layer_runtime(model_config=model_config)
        initialize_moe_config()
        install_layer_range(
            get_parallel().pp_group, pp_start_layer, pp_end_layer, num_hidden_layers
        )
        get_parallel().override_permanently(pp_size=VIRTUAL_PIPELINE_WIDTH)

        super().__init__(
            model_config=model_config,
            mem_fraction_static=mem_fraction_static,
            gpu_id=gpu_id,
            nccl_port=nccl_port,
            server_args=server_args,
        )

    def allocate_serving_state(self) -> None:
        """Allocate the KV pools, attention backends and graphs a constructed runner serves with.

        SGLang's runner only loads the model when it is constructed; its scheduler then allocates the
        pools, builds the attention backends, captures the graphs and warms the sampler, in this
        order. Parallax has no SGLang scheduler, so this performs the same steps for the one
        target runner.
        """
        self.account_preloaded_weights(self.preloaded_weights_bytes)
        self.alloc_memory_pool()
        self.init_attention_backends()
        self.init_cuda_graphs()
        prewarm_graph_pool_borrow()
        with torch.get_device_module(self.device).stream(self.forward_stream):
            self.prewarm_sampling()
        if self.token_to_kv_pool.post_capture_active:
            self.post_capture_resize_kv_pool()


def form_sgl_server_args(
    model_path: str,
    dtype: str = "bfloat16",
    kv_cache_memory_fraction: float = 0.85,
    tp_size: int = 1,
    dp_size: int = 1,
    attention_backend: str = "flashinfer",
    enable_dp_attention: bool = False,
    kv_block_size: int = 64,
    moe_runner_backend="auto",
    enable_lora: Optional[bool] = False,
    max_lora_rank: Optional[int] = None,
    lora_target_modules: Optional[List[str]] = None,
    lora_paths: Optional[List[str]] = None,
    max_loras_per_batch: Optional[int] = None,
    max_loaded_loras: Optional[int] = None,
    lora_eviction_policy: Optional[str] = "lru",
    lora_backend: Optional[str] = "triton",
    max_lora_chunk_size: Optional[int] = 128,
    max_num_tokens_per_batch: int = 16384,
    chunked_prefill_size: Optional[int] = None,
):
    """Creates a SGL ServerArgs object.

    chunked_prefill_size is None when the caller does not chunk prefills, which SGLang spells -1.
    """
    sgl_server_args = ServerArgs(
        model_path=model_path,
        dtype=dtype,
        attention_backend=attention_backend,
        enable_dp_attention=enable_dp_attention,
        page_size=kv_block_size,
        mem_fraction_static=kv_cache_memory_fraction,
        moe_runner_backend=moe_runner_backend,
        tp_size=tp_size,
        trust_remote_code=True,
        enable_lora=enable_lora,
        max_lora_rank=max_lora_rank,
        lora_target_modules=lora_target_modules,
        lora_paths=lora_paths,
        max_loras_per_batch=max_loras_per_batch,
        max_loaded_loras=max_loaded_loras,
        lora_eviction_policy=lora_eviction_policy,
        lora_backend=lora_backend,
        max_lora_chunk_size=max_lora_chunk_size,
        dp_size=dp_size,
        max_total_tokens=max_num_tokens_per_batch,
        chunked_prefill_size=-1 if chunked_prefill_size is None else chunked_prefill_size,
        # Parallax schedules batches itself; SGLang's overlap scheduler is never used.
        disable_overlap_schedule=True,
    )
    return sgl_server_args


def initialize_sgl_model_runner(
    model_repo: str,
    start_layer: int,
    end_layer: int,
    kv_cache_memory_fraction: float,
    attention_backend: str,
    kv_block_size: int,
    moe_runner_backend: str,
    max_num_tokens_per_batch: int = 16384,
    chunked_prefill_size: Optional[int] = None,
    enable_lora: Optional[bool] = False,
    max_lora_rank: Optional[int] = None,
    lora_target_modules: Optional[List[str]] = None,
    lora_paths: Optional[List[str]] = None,
    max_loras_per_batch: Optional[int] = None,
    max_loaded_loras: Optional[int] = None,
    lora_eviction_policy: Optional[str] = "lru",
    lora_backend: Optional[str] = "triton",
    max_lora_chunk_size: Optional[int] = 128,
    enable_dp_attention: bool = False,
    **kwargs,
):
    """
    Creates a SGL ModelRunner object.
    Returns:
      - model_runner: SGL model runner
      - config: model config driven by mlx-lm
      - tokenizer: tokenizer driven by mlx-lm
    """
    apply_parallax_sglang_monkey_patch()

    # Extract TP-related parameters from kwargs or use defaults
    tp_rank = kwargs.get("tp_rank", 0)
    tp_size = kwargs.get("tp_size", 1)
    dp_size = kwargs.get("dp_size", 1)
    use_hfcache = kwargs.get("use_hfcache", False)
    nccl_port = kwargs.get("nccl_port", None)
    # Use selective download for GPU models to save bandwidth and disk space
    from parallax.utils.model_download import selective_model_download

    logger.info(
        f"Downloading model with selective weight files for layers [{start_layer}, {end_layer})"
    )
    model_path = selective_model_download(
        model_repo, start_layer=start_layer, end_layer=end_layer, local_files_only=use_hfcache
    )

    config = normalize_model_config(load_config(model_path))
    tokenizer = load_tokenizer(model_path, eos_token_ids=config.get("eos_token_id", None))
    dtype = config.get("torch_dtype") or "bfloat16"

    if nccl_port is None:
        nccl_port = random.randint(4000, 5000)

    # Handling mxfp4 arguments
    quant_method = config.get("quant_method", None)
    quantization_config = config.get("quantization_config", None)
    if quant_method is None and quantization_config is not None:
        quant_method = quantization_config.get("quant_method", None)
    if quant_method == "mxfp4":
        attention_backend = "triton"
        moe_runner_backend = "triton_kernel"

    architectures = config.get("architectures", [])
    if architectures and any("Qwen3Next" in arch for arch in architectures):
        logger.debug("Qwen3-Next model detected, setting kv_block_size to 1")
        kv_block_size = 1

    server_args = form_sgl_server_args(
        str(model_path),
        dtype,
        kv_cache_memory_fraction,
        tp_size,
        dp_size,
        attention_backend,
        enable_dp_attention,
        kv_block_size,
        moe_runner_backend,
        enable_lora,
        max_lora_rank,
        lora_target_modules,
        lora_paths,
        max_loras_per_batch,
        max_loaded_loras,
        lora_eviction_policy,
        lora_backend,
        max_lora_chunk_size,
        max_num_tokens_per_batch=max_num_tokens_per_batch,
        chunked_prefill_size=chunked_prefill_size,
    )
    quant_method = None
    if (quantization_config := config.get("quantization_config", None)) is not None:
        quant_method = quantization_config.get("quant_method")
    model_config = ModelConfig(
        model_path=str(model_path),
        model_override_args="{}",
        dtype=dtype,
        quantization=quant_method,
    )
    # Only disable tie_word_embeddings when running a partial layer range
    # (multi-node PP where this node doesn't have both embed_tokens and lm_head).
    # For single-node or full-range runs, keep the original setting so that
    # lm_head correctly shares weights with embed_tokens.
    normalized_config = normalize_model_config(model_config.hf_config.to_dict())
    num_hidden_layers = normalized_config["num_hidden_layers"]
    model_config.hf_config.num_hidden_layers = num_hidden_layers
    if start_layer > 0 or end_layer < num_hidden_layers:
        model_config.hf_config.tie_word_embeddings = False
    model_config.hf_config.start_layer = start_layer
    model_config.hf_config.end_layer = end_layer
    if hasattr(model_config.hf_config, "text_config"):
        model_config.hf_config.text_config.num_hidden_layers = num_hidden_layers
        model_config.hf_config.text_config.start_layer = start_layer
        model_config.hf_config.text_config.end_layer = end_layer

    logger.debug(f"model_start_layer: {model_config.hf_config.start_layer}")
    logger.debug(f"model_end_layer: {model_config.hf_config.end_layer}")

    model_runner = ParallaxModelRunner(
        model_config=model_config,
        mem_fraction_static=kv_cache_memory_fraction,
        gpu_id=tp_rank,  # Currently reuse tp_rank to only support TP.
        nccl_port=nccl_port,
        server_args=server_args,
        pp_start_layer=start_layer,
        pp_end_layer=end_layer,
        tp_rank=tp_rank,
    )
    model_runner.allocate_serving_state()
    return model_runner, config, tokenizer


def refit_sgl_model(
    model_runner: ParallaxModelRunner,
    tensors: dict = None,
    refit_weight_path: str = None,
):
    """Runtime weight refit from disk"""
    if tensors is not None:
        logger.info("Executor begins weight refit from host memory")
        for x in tensors.keys():
            refit_tensors = [(x, tensors.get(x))]
            model_runner.update_weights_from_tensor(named_tensors=refit_tensors, load_format=None)
    elif refit_weight_path is not None:
        logger.info("Executor begins weight refit from disk files")
        model_runner.update_weights_from_disk(model_path=refit_weight_path, load_format="auto")
    else:
        assert False, "Weight refit needs host tensors or weight path"
    logger.info("Finish weight refit")
