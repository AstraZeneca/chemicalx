"""Base classes for models and utilities."""

import torch
from torch import nn
from torchdrug.data import PackedGraph

__all__ = [
    "Model",
    "ContextModel",
    "ContextlessModel",
]


class Model(nn.Module):
    """The base class for ChemicalX models."""


class ContextModel(Model):
    """A model that uses context features."""

    def forward(
        self,
        context_features: torch.FloatTensor,
        drug_features_left: torch.FloatTensor,
        drug_features_right: torch.FloatTensor,
    ) -> torch.FloatTensor:
        raise NotImplementedError


class ContextlessModel(Model):
    """A model that does not use context features."""

    def forward(self, molecules_left: PackedGraph, molecules_right: PackedGraph) -> torch.FloatTensor:
        """
        A forward pass of a contextless model.

        Args:
            molecules_left (torch.FloatTensor): Batched molecules for the left side drugs.
            molecules_right (torch.FloatTensor): Batched molecules for the right side drugs.
        Returns:
            hidden (torch.FloatTensor): A column vector of predicted scores.
        """
        raise NotImplementedError