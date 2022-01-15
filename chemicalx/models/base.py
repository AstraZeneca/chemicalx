"""Base classes for models and utilities."""

from abc import ABC, abstractmethod

from torch import nn

from chemicalx.data import DrugPairBatch

__all__ = [
    "UnimplementedModel",
    "Model",
]


class UnimplementedModel:
    """The base class for unimplemented ChemicalX models."""

    def __init__(self, x: int):
        """Instantiate a base model."""
        self.x = x


class Model(nn.Module, ABC):
    """The base class for ChemicalX models."""

    @abstractmethod
    def unpack(self, batch: DrugPairBatch):
        """Unpack a batch into a tuple of the features needed for forward.

        :param batch: A batch object
        :returns: A tuple that will be used as the positional arguments
            in this model's :func:`forward` method.
        """
