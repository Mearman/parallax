"""The first peer's side of speculative decoding across a split pipeline.

The first peer runs the input token and its drafts through its layers as one chunk and sends the rows after the
first down the pipeline. The last peer verifies them and returns the tokens it accepted and the next drafts. Tokens
the last peer rejected must not remain in this peer's state, so before each chunk this peer snapshots the request's
recurrent state, and when drafts were rejected it restores the snapshot and cuts its context back.

It does not run the accepted tokens again straight away. They are kept as `pending` and processed at the start of
the next chunk, ahead of the next input token: decoding on a quantised model is bound by reading the weights, so a
chunk a few tokens longer costs about what the shorter one did, while a separate replay would add a full pass to the
step that waits on this peer.
"""

from dataclasses import dataclass, field
from typing import Dict, List

from parallax.server.cache_manager import CacheManager


@dataclass
class _Request:
    # Tokens the last peer accepted that this peer's state does not include yet.
    pending: List[int] = field(default_factory=list)
    # The drafts the last peer proposed for the next step.
    drafts: List[int] = field(default_factory=list)
    # The step in flight: its chunk, the snapshot taken before it and the context length it started from.
    chunk: List[int] = field(default_factory=list)
    snapshot: int = -1
    absorbed: int = 0


@dataclass(frozen=True)
class Step:
    """What the first peer runs for one request this step."""

    tokens: List[int]  # the chunk: pending tokens, the input token, then the drafts
    prefix_len: int  # tokens of the request this peer's state already includes
    send_rows: int  # how many trailing rows of the chunk's output go down the pipeline: the input token and its drafts


class FirstPeerSpeculation:
    """Bookkeeping for the requests of a first peer that speculates across a split."""

    def __init__(self, cache_manager: CacheManager) -> None:
        self._cache = cache_manager
        self._requests: Dict[str, _Request] = {}

    def start(self, request_id: str, drafts: List[int]) -> None:
        """Begin speculating on a request whose prompt this peer has absorbed, with the drafts the last peer proposed."""
        self._requests[request_id] = _Request(drafts=list(drafts))

    def drafts(self, request_id: str) -> List[int]:
        return self._requests[request_id].drafts

    def begin(self, request_id: str, input_token: int) -> Step:
        """Reserve the cache for this step's chunk and snapshot the state it will change."""
        request = self._requests[request_id]
        assert not request.chunk, f"{request_id} already has a step in flight"
        tokens = request.pending + [input_token] + request.drafts
        request.absorbed = self._cache.get_context_length(request_id)
        if not self._cache.extend_slots(request_id, len(tokens)):
            raise RuntimeError(f"no cache room for a {len(tokens)} token chunk of {request_id}")
        request.snapshot = self._cache.snapshot_state(request_id)
        request.chunk = tokens
        return Step(tokens=tokens, prefix_len=request.absorbed, send_rows=1 + len(request.drafts))

    def finish(self, request_id: str, accepted: List[int], next_drafts: List[int]) -> None:
        """Apply the last peer's verdict: `accepted` is the drafts it accepted followed by the token it sampled."""
        request = self._requests[request_id]
        assert request.chunk, f"{request_id} has no step in flight"
        accepted_drafts = len(accepted) - 1
        draft_count = len(request.drafts)
        assert 0 <= accepted_drafts <= draft_count, f"{accepted_drafts} drafts accepted of {draft_count}"
        input_index = len(request.pending)
        if accepted_drafts == draft_count:
            request.pending = []
        else:
            self._cache.restore_state(request_id, request.snapshot)
            self._cache.truncate_context(request_id, request.absorbed)
            request.pending = request.chunk[: input_index + 1 + accepted_drafts]
        self._cache.release_snapshot(request.snapshot)
        request.snapshot = -1
        request.chunk = []
        request.drafts = list(next_drafts)

    def release(self, request_id: str) -> None:
        """Forget a request, returning a snapshot slot a step in flight still holds."""
        request = self._requests.pop(request_id, None)
        if request is not None and request.snapshot != -1:
            self._cache.release_snapshot(request.snapshot)
