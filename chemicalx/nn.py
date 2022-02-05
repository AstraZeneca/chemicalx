"""Various modules for chemicalx."""

import torch
import torch.nn.functional
from torch import nn

__all__ = [
    "Normalize",
]


class Normalize(nn.Module):
    """A modular wrapper around normlize()."""

    def __init__(self, p: float = 2.0, dim: int = 1, eps: float = 1e-12):
        super().__init__()
        self.p = p
        self.dim = dim
        self.eps = eps

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        return torch.nn.functional.normalize(x, p=self.p, dim=self.dim, eps=self.eps)
