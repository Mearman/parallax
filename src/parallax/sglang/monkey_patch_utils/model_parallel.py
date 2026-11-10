"""Parallax model-parallel patches for sglang.

Summary:
- ParallaxGroupCoordinator: a pipeline group whose first and last rank are decided by the layer
    range the process owns, installed on an existing group by install_layer_range.
- parallax_get_pp_indices: replaces sglang's get_pp_indices wherever it is imported, so the layers
    a stage builds and the layers its KV cache covers are the range recorded on the group.

These are minimal, reversible patches to support decentralized per-layer pipeline parallelism.
Remove when upstream sglang provides native support.
"""

import logging
from typing import Tuple

from sglang.srt.distributed.parallel_state import (
    GroupCoordinator as SGLGroupCoordinator,
)
from sglang.srt.distributed.parallel_state import get_pp_group

logger = logging.getLogger(__name__)



# Every Parallax process is one stage of a pipeline whose other stages live in other processes
# (and other engines), so SGLang sees a pipeline wider than the one process group it builds. Any
# width above one makes the model code and the graph runners take their pipeline path, which is
# the path that receives the previous stage's hidden states. The group is never used for
# collectives: Parallax moves activations between stages itself.
VIRTUAL_PIPELINE_WIDTH = 2


class ParallaxGroupCoordinator(SGLGroupCoordinator):
    """A pipeline group that knows which layers its process owns.

    A Parallax stage is first when it owns layer zero and last when it owns the final layer,
    whatever its rank in the process group, and the group reports the virtual pipeline width
    so that it agrees with the width published to SGLang's runtime context.
    """

    pp_start_layer: int
    pp_end_layer: int
    hidden_layers: int

    @property
    def world_size(self):
        """Return the virtual pipeline width, not the size of the one-process group."""
        return VIRTUAL_PIPELINE_WIDTH

    @world_size.setter
    def world_size(self, value):
        """Ignore the assignment made while SGLang builds the group."""

    @property
    def is_first_rank(self):
        """Return whether this stage owns the first layer."""
        return self.pp_start_layer == 0

    @property
    def is_last_rank(self):
        """Return whether this stage owns the final layer."""
        return self.pp_end_layer == self.hidden_layers


def install_layer_range(
    pp_group: SGLGroupCoordinator, pp_start_layer: int, pp_end_layer: int, hidden_layers: int
) -> None:
    """Record the owned layer range on an existing pipeline group.

    The group is retyped in place instead of being built through a patched constructor, so the
    patch does not depend on the signatures SGLang uses to create groups.
    """
    pp_group.__class__ = ParallaxGroupCoordinator
    pp_group.pp_start_layer = pp_start_layer
    pp_group.pp_end_layer = pp_end_layer
    pp_group.hidden_layers = hidden_layers


def parallax_get_pp_indices(
    num_hidden_layers: int, pp_rank: int, pp_size: int
) -> Tuple[int, int]:
    """Return the layer range this process owns instead of SGLang's even split.

    SGLang decides which layers a stage builds, and which layers its KV cache covers, by calling
    get_pp_indices. Parallax assigns ranges itself and records them on the pipeline group, so
    every caller must see that range, whatever pipeline rank and width it passes.
    """
    group = get_pp_group()
    return group.pp_start_layer, group.pp_end_layer


def apply_model_parallel_monkey_patch():
    """Make every SGLang caller of get_pp_indices see the layer range recorded on the group."""
    import sys

    import sglang.srt.distributed.utils as distributed_utils

    original = distributed_utils.get_pp_indices
    if original is parallax_get_pp_indices:
        return
    # Modules that imported the function by name keep their own reference, so replace each one.
    for module in list(sys.modules.values()):
        if getattr(module, "get_pp_indices", None) is original:
            module.get_pp_indices = parallax_get_pp_indices
