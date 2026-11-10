"""Prefilling a long prompt on a first peer in slices, with no prefix cache and nothing sent downstream until the end.

Attention over a whole long prompt needs a score buffer the size of the prompt squared, which is more than one Metal buffer
can hold. Parallax's own chunked prefill avoids that by finishing a slice, caching its blocks in the prefix cache and
sending the slice on; the next slice finds the previous ones through the cache. That cache skips prompt tokens a stage has
seen before, so a last stage without a prefix cache of its own would receive fewer rows than the prompt has tokens.

Here each slice continues the same request from the tokens already absorbed, as a later chunk of a chunked prefill does,
and the rows of every slice are joined, so the next stage receives one ordinary full prefill.
"""

from typing import Any, List

import mlx.core as mx

from parallax.server.cache_manager import CacheManager
from parallax.utils.utils import create_causal_mask


def continuation_inputs(
    cache_manager: CacheManager,
    request_ids: List[str],
    token_lists: List[Any],
    prefix_lens: List[int],
    dtype: mx.Dtype,
    pad_value: int,
) -> dict:
    """The arguments of a shard's forward pass for chunks that continue requests from `prefix_lens` absorbed tokens.

    `token_lists` hold token ids or hidden-state rows; the cache for the chunks must already be reserved. Chunks of different
    lengths are padded on the right.
    """
    from parallax.utils.utils import combine_padding_and_causal_masks, pad_inputs

    padded_inputs, padding_mask = pad_inputs(pad_value, token_lists, dtype)
    width = padded_inputs.shape[1]
    block_tables = [cache_manager.get_block_table(rid) for rid in request_ids]
    lengths = [len(tokens) if isinstance(tokens, list) else tokens.shape[0] for tokens in token_lists]

    slot_mapping_tensor = None
    if cache_manager.needs_blocks:
        block_size = cache_manager.block_size
        slot_mapping = []
        for table, prefix, length in zip(block_tables, prefix_lens, lengths):
            for index in range(width):
                if index < length:
                    position = prefix + index
                    slot_mapping.append(table[position // block_size] * block_size + position % block_size)
                else:
                    slot_mapping.append(-1)
        slot_mapping_tensor = mx.array(slot_mapping, dtype=mx.int64)

    max_blocks = max(len(table) for table in block_tables)
    block_tables_tensor = mx.array([table + [0] * (max_blocks - len(table)) for table in block_tables], dtype=mx.int32)
    mask = combine_padding_and_causal_masks(padding_mask, create_causal_mask(width, width, dtype), dtype)

    state_slot_mapping = None
    if cache_manager.needs_slots:
        state_slot_mapping = mx.array([cache_manager.get_slot(rid) for rid in request_ids], dtype=mx.int32)
    return {
        "h_or_tokens": padded_inputs,
        "cache": cache_manager.get_caches(),
        "mask": mask,
        "block_tables": block_tables_tensor,
        "context_lengths": mx.array([prefix + length for prefix, length in zip(prefix_lens, lengths)], dtype=mx.int32),
        "slot_mapping": slot_mapping_tensor,
        "state_slot_mapping": state_slot_mapping,
        "prefix_lens": mx.array(prefix_lens, dtype=mx.int32),
    }


def prefill_in_slices(
    model_shard: Any,
    cache_manager: CacheManager,
    request_id: str,
    tokens: List[int],
    slice_size: int,
    dtype: mx.Dtype,
    pad_value: int,
) -> mx.array:
    """Run a request's prompt through a shard `slice_size` tokens at a time and return all its rows, shape (1, len(tokens), hidden).

    The request's cache for the whole prompt must already be allocated.
    """
    if slice_size <= 0:
        raise ValueError(f"the slice size must be positive, got {slice_size}")
    rows = []
    for start in range(0, len(tokens), slice_size):
        chunk = tokens[start : start + slice_size]
        out = model_shard(
            **continuation_inputs(cache_manager, [request_id], [chunk], [start], dtype, pad_value)
        )
        cache_manager.materialize_linear_caches() if cache_manager.needs_slots else None
        mx.eval(out)
        rows.append(out[:, : len(chunk)])
    return mx.concatenate(rows, axis=1) if len(rows) > 1 else rows[0]
