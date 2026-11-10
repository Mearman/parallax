"""Rolling a request back after a rejected draft must leave it exactly as if the rejected tokens never ran.

A tiny random Qwen3.5-shaped model (three gated delta-net layers and one attention layer) processes a prompt and then a
verify chunk holding the committed token, one accepted draft and one rejected draft. The request is then rolled back to
the accepted tokens and the next token is decoded. Plain one-token decoding of the same tokens is the reference.
"""

import mlx.core as mx
import mlx.nn as nn
import pytest
from mlx_lm.models.qwen3_5 import TextModelArgs

from parallax.models.qwen3_5 import ParallaxQwen35Block
from parallax.server.cache_manager import CacheManager
from parallax.server.model import ShardedModel
from parallax.utils.utils import create_causal_mask

BLOCK_SIZE = 8
LAYERS = 4
VOCAB = 97


def _args() -> TextModelArgs:
    return TextModelArgs(
        model_type="qwen3_5_text",
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=LAYERS,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=32,
        linear_num_value_heads=2,
        linear_num_key_heads=1,
        linear_key_head_dim=16,
        linear_value_head_dim=16,
        linear_conv_kernel_dim=4,
        vocab_size=VOCAB,
        full_attention_interval=4,
        rope_parameters={"type": "default", "mrope_section": [2, 1, 1], "rope_theta": 10000, "partial_rotary_factor": 0.25},
    )


def _model() -> ShardedModel:
    mx.random.seed(7)
    model = ShardedModel(config=_args(), model_id="tiny", start_layer=0, end_layer=LAYERS, block_class=ParallaxQwen35Block, dtype=mx.float32)
    mx.eval(model.parameters())
    return model


def _cache() -> CacheManager:
    return CacheManager(
        num_layers=LAYERS,
        num_kv_heads=1,
        head_dim=32,
        dtype=mx.float32,
        block_size=BLOCK_SIZE,
        cache_memory_fraction=0.01,
        max_num_seqs=4,
        layer_types=["linear", "linear", "linear", "attention"],
        conv_dim=2 * 16 + 2 * 16,
        conv_kernel_size=4,
        linear_k_dim=16,
        linear_v_dim=16,
        linear_num_k_heads=1,
        linear_num_v_heads=2,
    )


def _run(model: ShardedModel, cm: CacheManager, rid: str, tokens: list[int], prefix_len: int) -> mx.array:
    """Process `tokens` after `prefix_len` committed tokens the way the executor's prefill does, and return their logits."""
    n = len(tokens)
    table = cm.get_block_table(rid)
    slots = [table[(prefix_len + i) // BLOCK_SIZE] * BLOCK_SIZE + (prefix_len + i) % BLOCK_SIZE for i in range(n)]
    logits = model(
        h_or_tokens=mx.array([tokens], dtype=mx.int32),
        cache=cm.get_caches(),
        mask=create_causal_mask(n, n, mx.float32),
        block_tables=mx.array([table], dtype=mx.int32),
        context_lengths=mx.array([prefix_len + n], dtype=mx.int32),
        slot_mapping=mx.array(slots, dtype=mx.int64),
        state_slot_mapping=mx.array([cm.get_slot(rid)], dtype=mx.int32),
        prefix_lens=mx.array([prefix_len], dtype=mx.int32),
    )
    cm.materialize_linear_caches()
    mx.eval(logits)
    return logits


PROMPT = [3, 14, 15, 92, 65, 35, 89, 79, 32, 38, 46]
COMMITTED, ACCEPTED, REJECTED, NEXT = 26, 43, 38, 33


def _reference() -> tuple[mx.array, mx.array]:
    """Plain decoding: the prompt, then COMMITTED, ACCEPTED and NEXT one token at a time."""
    model, cm = _model(), _cache()
    assert cm.allocate_request("r", len(PROMPT))[0]
    _run(model, cm, "r", PROMPT, 0)
    length = len(PROMPT)
    out = []
    for token in (COMMITTED, ACCEPTED, NEXT):
        assert cm.extend_slots("r", 1)
        out.append(_run(model, cm, "r", [token], length))
        length += 1
    return mx.concatenate(out, axis=1), mx.array(length)


def test_a_verify_chunk_matches_one_token_decoding_until_the_rejected_draft():
    reference, _ = _reference()
    model, cm = _model(), _cache()
    assert cm.allocate_request("r", len(PROMPT))[0]
    _run(model, cm, "r", PROMPT, 0)
    assert cm.extend_slots("r", 3)
    chunk = _run(model, cm, "r", [COMMITTED, ACCEPTED, REJECTED], len(PROMPT))
    assert mx.allclose(chunk[:, :2], reference[:, :2], atol=1e-4), "the accepted positions must not depend on the rejected draft"


def test_rolling_back_leaves_the_request_as_if_the_rejected_draft_never_ran():
    reference, _ = _reference()
    model, cm = _model(), _cache()
    assert cm.allocate_request("r", len(PROMPT))[0]
    _run(model, cm, "r", PROMPT, 0)

    snapshot = cm.snapshot_state("r")
    assert cm.extend_slots("r", 3)
    _run(model, cm, "r", [COMMITTED, ACCEPTED, REJECTED], len(PROMPT))

    # Two of the three tokens are kept: undo the chunk, then run the kept ones again.
    cm.restore_state("r", snapshot)
    cm.truncate_context("r", len(PROMPT))
    cm.release_snapshot(snapshot)
    assert cm.extend_slots("r", 2)
    replay = _run(model, cm, "r", [COMMITTED, ACCEPTED], len(PROMPT))
    assert mx.allclose(replay, reference[:, :2], atol=1e-4)

    assert cm.extend_slots("r", 1)
    after = _run(model, cm, "r", [NEXT], len(PROMPT) + 2)
    assert mx.allclose(after, reference[:, 2:3], atol=1e-4), "the token after a rollback must decode as in plain decoding"


def test_skipping_the_restore_changes_the_next_token():
    reference, _ = _reference()
    model, cm = _model(), _cache()
    assert cm.allocate_request("r", len(PROMPT))[0]
    _run(model, cm, "r", PROMPT, 0)
    assert cm.extend_slots("r", 3)
    _run(model, cm, "r", [COMMITTED, ACCEPTED, REJECTED], len(PROMPT))
    cm.truncate_context("r", len(PROMPT) + 2)  # the context shrinks but the recurrent state keeps the rejected token
    assert cm.extend_slots("r", 1)
    after = _run(model, cm, "r", [NEXT], len(PROMPT) + 2)
    assert not mx.allclose(after, reference[:, 2:3], atol=1e-4), "the check must be able to fail"


def test_truncating_returns_the_blocks_of_the_forgotten_tokens():
    model, cm = _model(), _cache()
    assert cm.allocate_request("r", BLOCK_SIZE)[0]
    free_before = cm.allocator.get_num_free_blocks()
    assert cm.extend_slots("r", BLOCK_SIZE + 1)
    assert cm.allocator.get_num_free_blocks() == free_before - 2
    cm.truncate_context("r", BLOCK_SIZE)
    assert cm.allocator.get_num_free_blocks() == free_before
    assert cm.get_context_length("r") == BLOCK_SIZE
    with pytest.raises(ValueError):
        cm.truncate_context("r", BLOCK_SIZE + 1)
