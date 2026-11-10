"""Per-phase timing of a request's steps, shared by the executors of every backend."""

import time
from typing import Dict, List


class StepTimings:
    """Where a request's speculative steps spend their time: the mean of each named phase over the steps that recorded it.

    The first peer records the time of its chunk, of the round trip to the last peer and back, and of applying the verdict; the
    last peer records its own step. The two logs then say what part of a step each machine and the link account for.
    """

    def __init__(self) -> None:
        self._seconds: Dict[str, Dict[str, List[float]]] = {}

    def record(self, request_id: str, phase: str, seconds: float) -> None:
        self._seconds.setdefault(request_id, {}).setdefault(phase, []).append(seconds)

    def clock(self) -> float:
        """The time source recorded durations are measured on."""
        return time.perf_counter()

    def summary(self, request_id: str) -> str:
        """A log line for a finished request, and forget it. Empty when the request recorded nothing."""
        phases = self._seconds.pop(request_id, None)
        if not phases:
            return ""
        parts = [
            f"{phase} {1000 * sum(values) / len(values):.1f} ms over {len(values)}"
            for phase, values in sorted(phases.items())
        ]
        return f"step timing request {request_id}: " + ", ".join(parts)
