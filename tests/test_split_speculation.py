"""A first peer that speculates across a split must generate exactly the tokens plain decoding does.

Two tiny random Qwen3.5-shaped shards stand for the first and last peers. The last peer verifies the drafts it is sent
by greedy comparison, rolls its own state back as vLLM does natively, and returns the accepted tokens and the next
drafts. The drafts are built from the plain-decoding answer with chosen mistakes, so full acceptance, partial acceptance
and total rejection all occur; the verdicts never trust a draft.
"""

import mlx.core as mx
from mlx_lm.models.qwen3_5 import TextModelArgs

from parallax.models.qwen3_5 import ParallaxQwen35Block
from parallax.server.cache_manager import CacheManager
from parallax.server.model import ShardedModel
from parallax.server.split_speculation import FirstPeerSpeculation
from parallax.utils.utils import create_causal_mask

BLOCK_SIZE = 8
LAYERS = 8
SPLIT = 4
VOCAB = 97
PROMPT = [3, 14, 15, 92, 65, 35, 89, 79, 32, 38, 46]
GENERATED = 24


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


def _shard(start: int, end: int, seed: int) -> ShardedModel:
    mx.random.seed(seed)
    model = ShardedModel(config=_args(), model_id="tiny", start_layer=start, end_layer=end, block_class=ParallaxQwen35Block, dtype=mx.float32)
    mx.eval(model.parameters())
    return model


def _cache(layer_types: list[str]) -> CacheManager:
    return CacheManager(
        num_layers=len(layer_types),
        num_kv_heads=1,
        head_dim=32,
        dtype=mx.float32,
        block_size=BLOCK_SIZE,
        cache_memory_fraction=0.01,
        max_num_seqs=4,
        layer_types=layer_types,
        conv_dim=2 * 16 + 2 * 16,
        conv_kernel_size=4,
        linear_k_dim=16,
        linear_v_dim=16,
        linear_num_k_heads=1,
        linear_num_v_heads=2,
    )


# Each shard holds one full-attention layer, as a real shard of this model does.
FIRST_TYPES = ["linear", "linear", "linear", "attention"]
LAST_TYPES = ["linear", "linear", "linear", "attention"]


def _run(model: ShardedModel, cm: CacheManager, rid: str, inputs: mx.array, prefix_len: int) -> mx.array:
    """Process a chunk after `prefix_len` absorbed tokens as the executor's prefill does; the chunk's cache room is reserved."""
    n = inputs.shape[1]
    table = cm.get_block_table(rid)
    slots = [table[(prefix_len + i) // BLOCK_SIZE] * BLOCK_SIZE + (prefix_len + i) % BLOCK_SIZE for i in range(n)]
    out = model(
        h_or_tokens=inputs,
        cache=cm.get_caches(),
        mask=create_causal_mask(n, n, mx.float32),
        block_tables=mx.array([table], dtype=mx.int32),
        context_lengths=mx.array([prefix_len + n], dtype=mx.int32),
        slot_mapping=mx.array(slots, dtype=mx.int64),
        state_slot_mapping=mx.array([cm.get_slot(rid)], dtype=mx.int32),
        prefix_lens=mx.array([prefix_len], dtype=mx.int32),
    )
    cm.materialize_linear_caches()
    mx.eval(out)
    return out


def _tokens(ids: list[int]) -> mx.array:
    return mx.array([ids], dtype=mx.int32)


def _greedy(logits: mx.array) -> list[int]:
    return [int(t) for t in mx.argmax(logits[0], axis=-1).tolist()]


def _plain_decoding() -> list[int]:
    first, last = _shard(0, SPLIT, 1), _shard(SPLIT, LAYERS, 2)
    cm_first, cm_last = _cache(FIRST_TYPES), _cache(LAST_TYPES)
    assert cm_first.allocate_request("r", len(PROMPT))[0] and cm_last.allocate_request("r", len(PROMPT))[0]
    hidden = _run(first, cm_first, "r", _tokens(PROMPT), 0)
    out = [_greedy(_run(last, cm_last, "r", hidden, 0))[-1]]
    length = len(PROMPT)
    while len(out) < GENERATED + 8:
        assert cm_first.extend_slots("r", 1) and cm_last.extend_slots("r", 1)
        hidden = _run(first, cm_first, "r", _tokens([out[-1]]), length)
        out.append(_greedy(_run(last, cm_last, "r", hidden, length))[-1])
        length += 1
    return out


class LastPeer:
    """The verifying peer: greedy verification with its own rollback, as vLLM does natively."""

    def __init__(self, truth: list[int], draft_length: int, mistakes) -> None:
        self.model = _shard(SPLIT, LAYERS, 2)
        self.cache = _cache(LAST_TYPES)
        self.truth = truth
        self.draft_length = draft_length
        self.mistakes = mistakes
        self.steps = 0

    def prefill(self, hidden: mx.array) -> tuple[int, list[int]]:
        assert self.cache.allocate_request("r", hidden.shape[1])[0]
        first = _greedy(_run(self.model, self.cache, "r", hidden, 0))[-1]
        return first, self._drafts_after(1)

    def _drafts_after(self, produced: int) -> list[int]:
        """Drafts for the tokens after the first `produced` of the answer, with this step's mistake applied."""
        drafts = list(self.truth[produced : produced + self.draft_length])
        position = self.mistakes(self.steps)
        wrong = range(len(drafts)) if position == "all" else [] if position is None else [position]
        for index in wrong:
            drafts[index] = (drafts[index] + 1) % VOCAB
        return drafts

    def verify(self, rows: mx.array, input_token: int, drafts: list[int], produced: int) -> tuple[list[int], list[int]]:
        length = self.cache.get_context_length("r")
        snapshot = self.cache.snapshot_state("r")
        assert self.cache.extend_slots("r", rows.shape[1])
        predictions = _greedy(_run(self.model, self.cache, "r", rows, length))
        accepted = 0
        while accepted < len(drafts) and drafts[accepted] == predictions[accepted]:
            accepted += 1
        tokens = predictions[: accepted + 1]
        if accepted < len(drafts):
            self.cache.restore_state("r", snapshot)
            self.cache.truncate_context("r", length)
            assert self.cache.extend_slots("r", accepted + 1)
            _run(self.model, self.cache, "r", rows[:, : accepted + 1], length)
        self.cache.release_snapshot(snapshot)
        self.steps += 1
        return tokens, self._drafts_after(produced + len(tokens))


def _speculative_decoding(truth: list[int], draft_length: int, mistakes) -> list[int]:
    first = _shard(0, SPLIT, 1)
    cm = _cache(FIRST_TYPES)
    last = LastPeer(truth, draft_length, mistakes)
    spec = FirstPeerSpeculation(cm)
    assert cm.allocate_request("r", len(PROMPT))[0]
    hidden = _run(first, cm, "r", _tokens(PROMPT), 0)
    token, drafts = last.prefill(hidden)
    out = [token]
    spec.start("r", drafts)
    while len(out) < GENERATED:
        step = spec.begin("r", out[-1])
        hidden = _run(first, cm, "r", _tokens(step.tokens), step.prefix_len)
        accepted, next_drafts = last.verify(hidden[:, -step.send_rows :], out[-1], spec.drafts("r"), len(out))
        spec.finish("r", accepted, next_drafts)
        out.extend(accepted)
    return out[:GENERATED]


def _mistakes(draft_length: int):
    """No mistake, then one in the first, the middle and the last draft position, then every position wrong, repeating."""
    cycle = [None, 0, draft_length // 2, draft_length - 1, "all"]

    def at(step: int):
        return cycle[step % len(cycle)]

    return at


def test_speculating_across_the_split_generates_what_plain_decoding_does():
    truth = _plain_decoding()
    for draft_length in (1, 3):
        assert _speculative_decoding(truth, draft_length, _mistakes(draft_length)) == truth[:GENERATED], draft_length


def test_every_snapshot_slot_is_returned():
    truth = _plain_decoding()
    first, cm = _shard(0, SPLIT, 1), _cache(FIRST_TYPES)
    last = LastPeer(truth, 3, _mistakes(3))
    spec = FirstPeerSpeculation(cm)
    assert cm.allocate_request("r", len(PROMPT))[0]
    token, drafts = last.prefill(_run(first, cm, "r", _tokens(PROMPT), 0))
    spec.start("r", drafts)
    free = cm.slot_allocator.get_num_free_slots()
    step = spec.begin("r", token)
    hidden = _run(first, cm, "r", _tokens(step.tokens), step.prefix_len)
    accepted, next_drafts = last.verify(hidden[:, -step.send_rows :], token, spec.drafts("r"), 1)
    spec.finish("r", accepted, next_drafts)
    assert cm.slot_allocator.get_num_free_slots() == free


def test_prefilling_in_slices_gives_the_rows_of_prefilling_at_once():
    from parallax.server.inline_prefill import prefill_in_slices

    prompt = PROMPT + list(range(60, 90))
    whole_model, whole_cache = _shard(0, SPLIT, 1), _cache(FIRST_TYPES)
    assert whole_cache.allocate_request("r", len(prompt))[0]
    whole = _run(whole_model, whole_cache, "r", _tokens(prompt), 0)
    for slice_size in (1, 7, 16, len(prompt), len(prompt) + 5):
        model, cm = _shard(0, SPLIT, 1), _cache(FIRST_TYPES)
        assert cm.allocate_request("r", len(prompt))[0]
        rows = prefill_in_slices(model, cm, "r", prompt, slice_size, mx.float32, 0)
        assert rows.shape == whole.shape, slice_size
        assert mx.allclose(rows, whole, atol=1e-4), slice_size


def test_a_slice_size_that_is_not_positive_is_refused():
    import pytest
    from parallax.server.inline_prefill import prefill_in_slices

    with pytest.raises(ValueError):
        prefill_in_slices(_shard(0, SPLIT, 1), _cache(FIRST_TYPES), "r", PROMPT, 0, mx.float32, 0)
