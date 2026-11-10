"""Reading the tokens a speculative decoding step produced."""

from typing import Any, Callable, List

from vllm.v1.sample.rejection_sampler import PLACEHOLDER_TOKEN_ID


def accepted_token_ids_of(sampled_token_ids: Any) -> List[int]:
    """The tokens one request's step accepted, in order.

    A step without speculation yields one id. A speculative step yields up to the draft length plus one, and vLLM fills the
    positions after the first rejection with PLACEHOLDER_TOKEN_ID, so the accepted tokens are the ones that are not that.
    """
    accepted = [int(token) for token in sampled_token_ids.reshape(-1).tolist() if token != PLACEHOLDER_TOKEN_ID]
    assert accepted, "a step always accepts at least one token"
    return accepted


def accepted_token_ids(request: Any) -> List[int]:
    """The tokens a request's step accepted: its sampled row if it carries one, else its single next token."""
    if request.hidden_states is None:
        return [request.next_token_id]
    return accepted_token_ids_of(request.hidden_states)


def commit_accepted(request: Any, token_ids: List[int], finishes: Callable[[Any], bool]) -> List[int]:
    """Commit a step's accepted tokens to the request in order and return the ones committed.

    The first token that finishes the request (`finishes` is true after committing it) ends the step, so tokens the
    drafter proposed beyond an end-of-sequence or the length limit are never committed.
    """
    committed: List[int] = []
    for token_id in token_ids:
        request.commit_new_token(token_id)
        committed.append(token_id)
        if finishes(request):
            break
    return committed
