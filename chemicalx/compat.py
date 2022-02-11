"""A compatibility layer for chemicalx."""

import torch
import torchdrug.data
from torch.types import Device

__all__ = [
    "PackedGraph",
    "Graph",
]


class PackedGraph(torchdrug.data.PackedGraph):
    """A compatibility later that implements a to() function.

    This can be removed when https://github.com/DeepGraphLearning/torchdrug/pull/70
    is merged and a new version of torchdrug is released.
    """

    def to(self, device: Device):
        """Return a copy of this packed graph on the given device."""
        if isinstance(device, str):
            if device == "cpu":
                return self.cpu()
            elif device == "cuda":
                return self.cuda()
            else:
                raise NotImplementedError(f"{self.__class__.__name__}.to() is not implemented for string: {device}")
        elif isinstance(device, torch.device):
            if device.type == "cpu":
                return self.cpu()
            elif device.type == "cuda":
                return self.cuda()
            else:
                raise NotImplementedError
        else:
            raise TypeError


class Graph(torchdrug.data.Graph):
    """A compatibility layer that makes appropriate packed graphs."""

    packed_type = PackedGraph
