"""The MLX executor's inline prefill path: a prefill batch of long prompts run in slices gives the rows a whole-prompt prefill gives."""

import types

import mlx.core as mx

from parallax.server.executor.mlx_executor import MLXExecutor
from parallax.server.request import InitialRequest
from parallax.server.sampling.sampling_params import SamplingParams
from tests.test_split_speculation import FIRST_TYPES, SPLIT, _cache, _run, _shard, _tokens

PROMPTS = {"a": list(range(3, 40)), "b": list(range(50, 71))}


def _requests() -> list[InitialRequest]:
    return [InitialRequest(request_id=rid, input_ids=ids, sampling_params=SamplingParams(), max_new_tokens=4) for rid, ids in PROMPTS.items()]


def test_the_inline_batch_gives_each_request_the_rows_of_its_whole_prompt():
    model, cm = _shard(0, SPLIT, 1), _cache(FIRST_TYPES)
    executor = types.SimpleNamespace(
        model_shard=model, cache_manager=cm, dtype=mx.float32, pad_token_id=0, inline_prefill_chunk_size=8
    )
    prepared = MLXExecutor._prepare_inline_prefill_batch(executor, _requests())
    assert prepared["context_lengths"].tolist() == [37, 21]
    rows = MLXExecutor._prefill_inline(executor, prepared)
    assert rows.shape[:2] == (2, 37), "padded on the right to the longest prompt"

    for index, (rid, prompt) in enumerate(PROMPTS.items()):
        reference_model, reference_cache = _shard(0, SPLIT, 1), _cache(FIRST_TYPES)
        assert reference_cache.allocate_request(rid, len(prompt))[0]
        whole = _run(reference_model, reference_cache, rid, _tokens(prompt), 0)
        assert mx.allclose(rows[index : index + 1, : len(prompt)], whole, atol=1e-4), rid
    assert not rows[1, 21:].any(), "the padding is zeros"
