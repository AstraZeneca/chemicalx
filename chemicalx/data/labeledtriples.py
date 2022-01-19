"""A module for the labeled triples class."""

from typing import List, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

__all__ = ["LabeledTriples"]


class LabeledTriples:
    """Labeled triples for drug pair scoring."""

    def __init__(self):
        """Initialize the labeled triples object."""
        self.columns = ["drug_1", "drug_2", "context", "label"]
        self.types = {"drug_1": str, "drug_2": str, "context": str, "label": float}
        self.data = pd.DataFrame(columns=self.columns).astype(self.types)

    def __len__(self) -> int:
        """Get the number of triples."""
        return len(self.data.index)

    def drop_duplicates(self):
        """Drop the duplicated entries."""
        self.data = self.data.drop_duplicates()

    def update_from_pandas(self, data: pd.DataFrame):
        """
        Update the labeled triples from a dataframe.

        Args:
            data (pd.DataFrame): A dataframe of labeled triples.
        """
        self.data = pd.concat([self.data, data])

    def update_from_list(self, data: List[List]):
        """
        Update the labeled triples from a list.

        Args:
            data (list): A list of labeled triples.
        """
        data = pd.DataFrame(data, columns=self.columns)
        self.data = pd.concat([self.data, data])

    def __add__(self, value):
        """
        Add the triples in two LabeledTriples objects together - syntactic sugar for '+'.

        Args:
            value (LabeledTriples): Another LabeledTriples object for the addition.
        Returns:
            new_triples (LabeledTriples): A LabeledTriples object after the addition.
        """
        new_triples = LabeledTriples()
        new_triples.update_from_pandas(pd.concat([self.data, value.data]))
        return new_triples

    def get_drug_count(self) -> int:
        """
        Get the number of drugs in the labeled triples dataset.

        Returns
            int: The number of unique compounds in the labeled triples dataset.
        """
        return pd.unique(self.data[["drug_1", "drug_2"]].values.ravel("K")).shape[0]

    def get_context_count(self) -> int:
        """
        Get the number of unique contexts in the labeled triples dataset.

        Returns
            int: The number of unique contexts in the labeled triples dataset.
        """
        return self.data["context"].nunique()

    def get_combination_count(self) -> int:
        """
        Get the number of unique drug pairs in the labeled triples dataset.

        Returns
            int: The number of unique pairs in the labeled triples dataset.
        """
        combination_count = self.data[["drug_1", "drug_2"]].drop_duplicates().shape[0]
        return combination_count

    def get_labeled_triple_count(self) -> int:
        """
        Get the number of triples in the labeled triples dataset.

        Returns
            int: The number of triples in the labeled triples dataset.
        """
        triple_count = self.data.shape[0]
        return triple_count

    def get_positive_count(self) -> int:
        """
        Get the number of positive triples in the dataset.

        Returns
            int: The number of positive triples.
        """
        return int(self.data["label"].sum())

    def get_negative_count(self) -> int:
        """
        Get the number of negative triples in the dataset.

        Returns
            int: The number of negative triples.
        """
        return self.get_labeled_triple_count() - self.get_positive_count()

    def get_positive_rate(self) -> float:
        """
        Get the ratio of positive triples in the dataset.

        Returns
            float: The ratio of positive triples.
        """
        return self.data["label"].mean()

    def get_negative_rate(self) -> float:
        """
        Get the ratio of positive triples in the dataset.

        Returns
            float: The ratio of negative triples.
        """
        return 1.0 - self.data["label"].mean()

    def train_test_split(self, train_size: float = 0.8, random_state: int = 42) -> Tuple:
        """
        Split the LabeledTriples object for training and testing.

        Args:
            train_size (float): The ratio of training triples. Default is 0.8.
            random_state (int): The random seed. Default is 42.
        Returns
            train_labeled_triples (LabeledTriples): The training triples.
            test_labeled_triples (LabeledTriples): The testing triples.
        """
        train_data, test_data = train_test_split(self.data, train_size=train_size, random_state=random_state)
        train_labeled_triples = LabeledTriples()
        test_labeled_triples = LabeledTriples()
        train_labeled_triples.update_from_pandas(train_data)
        test_labeled_triples.update_from_pandas(test_data)
        return train_labeled_triples, test_labeled_triples
