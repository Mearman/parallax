"""Reading the tokens a speculative decoding step produced, for the vLLM backend."""

from typing import List

import torch
from vllm.v1.sample.rejection_sampler import PLACEHOLDER_TOKEN_ID

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
