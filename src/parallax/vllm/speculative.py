"""Reading the tokens a speculative decoding step produced, for the vLLM backend."""

from typing import List

import torch
from vllm.model_executor.models.utils import PPMissingLayer
from vllm.v1.sample.rejection_sampler import PLACEHOLDER_TOKEN_ID
from vllm.v1.spec_decode.llm_base_proposer import SpecDecodeBaseProposer

from parallax.server.speculation import (  # noqa: F401  (re-exported for the backend's callers)
    PLACEHOLDER_TOKEN_ID as SHARED_PLACEHOLDER_TOKEN_ID,
)
from parallax.server.speculation import (  # noqa: F401
    SpeculationStats,
    accepted_token_ids,
    accepted_token_ids_of,
    commit_accepted,
)

assert PLACEHOLDER_TOKEN_ID == SHARED_PLACEHOLDER_TOKEN_ID, "vLLM changed the value that marks a rejected draft position"


def pad_sampled_token_ids(rows: List[List[int]], width: int) -> torch.Tensor:
    """One row per request of the tokens its step accepted, padded on the right with PLACEHOLDER_TOKEN_ID to `width`.

    Without async scheduling vLLM returns the accepted tokens as lists of different lengths; the rest of Parallax reads one
    fixed-width row per request, the shape the async output had.
    """
    padded = torch.full((len(rows), width), PLACEHOLDER_TOKEN_ID, dtype=torch.int64)
    for index, row in enumerate(rows):
        assert len(row) <= width, f"a step accepted {len(row)} tokens but the row is {width} wide"
        if row:
            padded[index, : len(row)] = torch.tensor(row, dtype=torch.int64)
    return padded


_original_share_embeddings = SpecDecodeBaseProposer._maybe_share_embeddings


def _share_embeddings_unless_missing(self: SpecDecodeBaseProposer, target_language_model: torch.nn.Module) -> None:
    """Share the target's embedding table with a multi-token-prediction drafter unless this stage has none.

    A stage that does not hold the first layers has no embedding table, only a placeholder, and sharing would replace the
    drafter's own copy, which it loads from the checkpoint, with that placeholder.
    """
    embed_tokens = getattr(getattr(target_language_model, "model", None), "embed_tokens", None)
    if isinstance(embed_tokens, PPMissingLayer):
        return
    _original_share_embeddings(self, target_language_model)


def keep_the_drafters_own_embeddings_on_stages_without_them() -> None:
    """Install the guard above on vLLM's proposer. Calling it again changes nothing."""
    SpecDecodeBaseProposer._maybe_share_embeddings = _share_embeddings_unless_missing
