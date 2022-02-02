"""A module for the context feature set class."""

from collections import UserDict
from typing import Iterable, Mapping, Sequence

import torch

__all__ = [
    "ContextFeatureSet",
]


class ContextFeatureSet(UserDict, Mapping[str, torch.FloatTensor]):
    """Context feature set for biological/chemical context feature vectors."""

    @classmethod
    def from_dict(cls, data: Mapping[str, Sequence[float]]) -> "ContextFeatureSet":
        """Generate a context feature set from a data dictionary."""
        return cls({key: torch.FloatTensor(values).view(1, -1) for key, values in data.items()})

    def get_feature_matrix(self, contexts: Iterable[str]) -> torch.FloatTensor:
        """Get the feature matrix for a list of contexts.

        :param contexts: A list of context identifiers.
        :returns: A matrix of context features.
        """
        return torch.cat([self.data[context] for context in contexts])
