import torch
import numpy as np
from typing import Dict, List


class ContextFeatureSet(dict):
    """
    Context feature set for biological/chemical context feature vectors.
    """

    def __setitem__(self, context: str, features: np.ndarray):
        """Setting the feature vector for a biological context key.

        Args:
            context (str): Biological or chemical context identifier.
            features (np.ndarray): Feature vector for the context.
        """
        self.__dict__[context] = torch.FloatTensor(features)

    def __getitem__(self, context: str) -> torch.FloatTensor:
        """Getting the feature vector for a biological context key.

        Args:
            context (str): Biological or chemical context identifier.
        Returns:
            torch.FloatTensor: The feature vector corresponding to the key.
        """
        return self.__dict__[context]

    def __len__(self):
        """Getting the number of biological/chemical contexts.

        Returns:
            int: The number of contexts.
        """
        return len(self.__dict__)

    def __delitem__(self, context: str):
        """Deleting the feature vector for a biological context key.

        Args:
            context (str): Biological or chemical context identifier.
        """
        del self.__dict__[context]

    def clear(self):
        """Deleting all of the contexts from the context feature set.

        Returns:
            ContextFeatureSet: An empty context feature set.
        """
        return self.__dict__.clear()

    def has_context(self, context: str):
        """Checking whether a context feature set contains a context.

        Args:
            context (str): Biological or chemical context identifier.
        Returns:
            bool: Boolean describing whether the context is in the context set.
        """
        return context in self.__dict__

    def update(self, data: Dict[str, np.ndarray]):
        """Updating a dictionary of context keys - feature vector values to a context set.

        Args:
            data (dict): A dictionary of context keys with feature vector values.
        Returns:
            ContextFeatureSet: The updated context feature set.
        """
        return self.__dict__.update({context: torch.FloatTensor(features) for context, features in data.items()})

    def keys(self):
        """Retrieving the list of biological / chemical contexts in a feature set.

        Returns:
            list: An iterator of context identifiers.
        """
        return self.__dict__.keys()

    def values(self):
        """Retrieving the iterator of context feature vectors.

        Returns:
            list: Feature vector iterator.
        """
        return self.__dict__.values()

    def items(self):
        """Retrieving the iterator of tuples containing context identifier - feature vector pairs.

        Returns:
            list: An iterator of (context - feature vector) tuples.
        """
        return self.__dict__.items()

    def __contains__(self, context: str):
        """A data class method which allows the use of the 'in' operator.

        Args:
            context (str): A context identifier.
        Returns:
            bool: An indicator whether the context is in the context feature set.
        """
        return context in self.__dict__

    def __iter__(self):
        """A data class method which allows iteration over the context feature set.

        Returns:
            iterable: An iterable of the context feature set.
        """
        return iter(self.__dict__)

    def get_context_count(self) -> int:
        """Getting the number of biological contexts.

        Returns:
            int: The number of contexts.
        """
        return len(self.__dict__)

    def get_feature_matrix(self, contexts: List[str]) -> torch.FloatTensor:
        """Getting the feature matrix for a list of contexts.

        Args:
            contexts (list): A list of context identifiers.
        Return:
            features (torch.FloatTensor): A matrix of context features.
        """
        features = torch.cat([self.__dict__[context] for context in contexts])
        return features
