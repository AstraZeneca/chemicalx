"""A module for the context feature set class."""

from collections import UserDict
from typing import Iterable, Mapping

import torch

__all__ = [
    "ContextFeatureSet",
]


class ContextFeatureSet(UserDict, Mapping[str, torch.FloatTensor]):
    """Context feature set for biological/chemical context feature vectors."""

    def get_feature_matrix(self, contexts: Iterable[str]) -> torch.FloatTensor:
        """Get the feature matrix for a list of contexts.

        Args:
            contexts: A list of context identifiers.
        Return:
            features: A matrix of context features.
        """
        return torch.cat([self.data[context] for context in contexts])
