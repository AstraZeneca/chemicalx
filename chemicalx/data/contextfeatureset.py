"""A module for the context feature set class."""

from typing import Dict, Iterable

import numpy as np
import torch

__all__ = [
    "ContextFeatureSet",
]


class ContextFeatureSet(dict):
    """Context feature set for biological/chemical context feature vectors."""

    def __setitem__(self, context: str, features: np.ndarray) -> None:
        """Set the feature vector for a biological context key.

        Args:
            context: Biological or chemical context identifier.
            features: Feature vector for the context.
        """
        self.__dict__[context] = torch.FloatTensor(features)

    def __getitem__(self, context: str) -> torch.FloatTensor:
        """Get the feature vector for a biological context key.

        Args:
            context: Biological or chemical context identifier.
        Returns:
            : The feature vector corresponding to the key.
        """
        return self.__dict__[context]

    def __len__(self) -> int:
        """Get the number of biological/chemical contexts.

        Returns:
            : The number of contexts.
        """
        return len(self.__dict__)

    def __delitem__(self, context: str) -> None:
        """Delete the feature vector for a biological context key.

        Args:
            context: Biological or chemical context identifier.
        """
        del self.__dict__[context]

    def clear(self) -> "ContextFeatureSet":
        """Delete all the contexts from the context feature set.

        Returns:
            : An empty context feature set.
        """
        return self.__dict__.clear()

    def has_context(self, context: str) -> bool:
        """Check whether a context feature set contains a context.

        Args:
            context: Biological or chemical context identifier.
        Returns:
            : Boolean describing whether the context is in the context set.
        """
        return context in self.__dict__

    def update(self, data: Dict[str, np.ndarray]):
        """Update a dictionary of context keys - feature vector values to a context set.

        Args:
            data (dict): A dictionary of context keys with feature vector values.
        Returns:
            ContextFeatureSet: The updated context feature set.
        """
        return self.__dict__.update({context: torch.FloatTensor(features) for context, features in data.items()})

    def keys(self):
        """Get the list of biological / chemical contexts in a feature set.

        Returns:
            list: An iterator of context identifiers.
        """
        return self.__dict__.keys()

    def values(self):
        """Get the iterator of context feature vectors.

        Returns:
            list: Feature vector iterator.
        """
        return self.__dict__.values()

    def items(self):
        """Get the iterator of tuples containing context identifier - feature vector pairs.

        Returns:
            list: An iterator of (context - feature vector) tuples.
        """
        return self.__dict__.items()

    def __contains__(self, context: str) -> bool:
        """Check if the context is in keys.

        Args:
            context (str): A context identifier.
        Returns:
            bool: An indicator whether the context is in the context feature set.
        """
        return context in self.__dict__

    def __iter__(self):
        """Iterate over the context feature set.

        Returns:
            iterable: An iterable of the context feature set.
        """
        return iter(self.__dict__)

    def get_context_count(self) -> int:
        """Get the number of biological contexts.

        Returns:
            : The number of contexts.
        """
        return len(self.__dict__)

    def get_feature_matrix(self, contexts: Iterable[str]) -> torch.FloatTensor:
        """Get the feature matrix for a list of contexts.

        Args:
            contexts: A list of context identifiers.
        Return:
            features: A matrix of context features.
        """
        features = torch.cat([self.__dict__[context] for context in contexts])
        return features
