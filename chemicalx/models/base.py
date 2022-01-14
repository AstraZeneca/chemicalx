"""Base classes for models and utilities."""

import torch
from torch import nn
from torchdrug.data import PackedGraph

__all__ = [
    "Model",
    "ContextModel",
    "ContextlessModel",
]


class UnimplementedModel:
    """The base class for unimplemnted ChemicalX models."""

    def __init__(self, x):
        self.x = x


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
        """
        A forward pass of a model.

        :param context_features: A matrix of biological context features.
        :param drug_features_left: A matrix of head drug features.
        :param drug_features_right: A matrix of tail drug features.
        :returns: A column vector of predicted scores.
        """
        raise NotImplementedError


class ContextlessModel(Model):
    """A model that does not use context features."""

    def forward(self, molecules_left: PackedGraph, molecules_right: PackedGraph) -> torch.FloatTensor:
        """
        A forward pass of a contextless model.

        :param molecules_left: Batched molecules for the left side drugs.
        :param molecules_right: Batched molecules for the right side drugs.
        :returns: A column vector of predicted scores.
        """
        raise NotImplementedError
