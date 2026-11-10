"""Speculative decoding steps that cross the pipeline: reading the tokens a step accepted and committing them.

Backend neutral, so the first peer can be an MLX executor and the last a vLLM one.
"""

from typing import Any, Callable, Dict, List

# vLLM fills the positions after a step's first rejection with this value; a sampled row sent between peers keeps it.
PLACEHOLDER_TOKEN_ID = -1


def accepted_token_ids_of(sampled_token_ids: Any) -> List[int]:
    """The tokens one request's step accepted, in order.

    A step without speculation yields one id. A speculative step yields up to the draft length plus one, and the
    positions after the first rejection hold PLACEHOLDER_TOKEN_ID, so the accepted tokens are the ones that are not that.
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


class SpeculationStats:
    """What speculation did for each request: its steps, the tokens they accepted, and how many of the drafted slots were accepted.

    A step that scheduled d draft slots and accepted n tokens accepted n - 1 drafts (the first token is the one the step
    would have produced anyway).
    """

    def __init__(self) -> None:
        self._steps: Dict[str, List[int]] = {}

    def record(self, request_id: str, draft_slots: int, accepted: int) -> None:
        steps = self._steps.setdefault(request_id, [0, 0, 0, 0])
        steps[0] += 1
        steps[1] += accepted
        steps[2] += draft_slots
        steps[3] += accepted - 1

    def summary(self, request_id: str) -> str:
        """A log line for a finished request, and forget it. Empty when the request recorded no step."""
        steps = self._steps.pop(request_id, None)
        if steps is None:
            return ""
        step_count, tokens, draft_slots, accepted_drafts = steps
        return (
            f"speculation request {request_id}: {tokens} tokens in {step_count} steps, "
            f"{accepted_drafts} of {draft_slots} drafted slots accepted"
        )
