"""Parallax model-parallel patches for sglang.

Summary:
- ParallaxGroupCoordinator: a pipeline group whose first and last rank are decided by the layer
    range the process owns, installed on an existing group by install_layer_range.
- monkey_patch_make_layers: replaces sglang.srt.utils.make_layers; uses
    get_pp_group().pp_start_layer/end_layer to instantiate local layers and PPMissingLayer
    placeholders for non-local layers.

These are minimal, reversible patches to support decentralized per-layer pipeline parallelism.
Remove when upstream sglang provides native support.
"""

import logging
from typing import Any, Dict, Optional, Tuple

import sglang.srt.utils
import torch
from sglang.srt.distributed.parallel_state import (
    GroupCoordinator as SGLGroupCoordinator,
)
from sglang.srt.utils import LayerFn, add_prefix


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


def monkey_patch_make_layers(
    num_hidden_layers: int,
    layer_fn: LayerFn,
    pp_rank: Optional[int] = None,
    pp_size: Optional[int] = None,
    prefix: str = "",
    return_tuple: bool = False,
    offloader_kwargs: Optional[Dict[str, Any]] = None,
) -> Tuple[torch.nn.ModuleList, int, int]:
    """A monkey patch to replace sglang.srt.utils.make_layers"""
    # circula imports
    from sglang.srt.distributed.parallel_state import get_pp_group
    from sglang.srt.layers.utils import PPMissingLayer
    from sglang.srt.utils.offloader import get_offloader

    assert not pp_size or num_hidden_layers >= pp_size
    start_layer, end_layer = get_pp_group().pp_start_layer, get_pp_group().pp_end_layer

    modules = torch.nn.ModuleList(
        [PPMissingLayer(return_tuple=return_tuple) for _ in range(start_layer)]
        + get_offloader().wrap_modules(
            (
                layer_fn(idx=idx, prefix=add_prefix(idx, prefix))
                for idx in range(start_layer, end_layer)
            ),
            **(offloader_kwargs or {}),
        )
        + [PPMissingLayer(return_tuple=return_tuple) for _ in range(end_layer, num_hidden_layers)]
    )
    if pp_rank is None or pp_size is None:
        return modules
    return modules, start_layer, end_layer


def apply_model_parallel_monkey_patch():
    sglang.srt.utils.make_layers = monkey_patch_make_layers
    import sglang.srt.utils.common as utils_common

    utils_common.make_layers = monkey_patch_make_layers
