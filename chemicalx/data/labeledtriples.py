"""A module for the labeled triples class."""

from typing import ClassVar, Iterable, Mapping, Optional, Sequence, Tuple, Union

import pandas as pd
from sklearn.model_selection import train_test_split

__all__ = ["LabeledTriples"]


class LabeledTriples:
    """Labeled triples for drug pair scoring."""

    columns: ClassVar[Sequence[str]] = ("drug_1", "drug_2", "context", "label")
    dtype: ClassVar[Mapping[str, type]] = {"drug_1": str, "drug_2": str, "context": str, "label": float}

    def __init__(self, data: Union[pd.DataFrame, Iterable[Sequence]]):
        """Initialize the labeled triples object."""
        if not isinstance(data, pd.DataFrame):
            data = pd.DataFrame(data, columns=self.columns).astype(self.dtype)
        self.data = data

    def __len__(self) -> int:
        """Get the number of triples."""
        return len(self.data.index)

    def drop_duplicates(self):
        """Drop the duplicated entries."""
        self.data = self.data.drop_duplicates()

    def __add__(self, value: "LabeledTriples") -> "LabeledTriples":
        """
        Add the triples in two LabeledTriples objects together - syntactic sugar for '+'.

        :param value: Another LabeledTriples object for the addition.
        :returns: A LabeledTriples object after the addition.
        """
        return LabeledTriples(pd.concat([self.data, value.data]))

    def get_drug_count(self) -> int:
        """Get the number of drugs in the labeled triples dataset."""
        return pd.unique(self.data[["drug_1", "drug_2"]].values.ravel("K")).shape[0]

    def get_context_count(self) -> int:
        """Get the number of unique contexts in the labeled triples dataset."""
        return self.data["context"].nunique()

    def get_combination_count(self) -> int:
        """Get the number of unique drug pairs in the labeled triples dataset."""
        combination_count = self.data[["drug_1", "drug_2"]].drop_duplicates().shape[0]
        return combination_count

    def get_labeled_triple_count(self) -> int:
        """Get the number of triples in the labeled triples dataset."""
        triple_count = self.data.shape[0]
        return triple_count

    def get_positive_count(self) -> int:
        """Get the number of positive triples in the dataset."""
        return int(self.data["label"].sum())

    def get_negative_count(self) -> int:
        """Get the number of negative triples in the dataset."""
        return self.get_labeled_triple_count() - self.get_positive_count()

    def get_positive_rate(self) -> float:
        """Get the ratio of positive triples in the dataset."""
        return self.data["label"].mean()

    def get_negative_rate(self) -> float:
        """Get the ratio of positive triples in the dataset."""
        return 1.0 - self.data["label"].mean()

    def train_test_split(
        self, train_size: Optional[float] = None, random_state: Optional[int] = 42
    ) -> Tuple["LabeledTriples", "LabeledTriples"]:
        """
        Split the LabeledTriples object for training and testing.

        :param train_size: The ratio of training triples. Default is 0.8 if None is passed.
        :param random_state: The random seed. Default is 42. Set to none for no fixed seed.
        :returns: A pair of training triples and testing triples
        """
        train_data, test_data = train_test_split(self.data, train_size=train_size or 0.8, random_state=random_state)
        return LabeledTriples(train_data), LabeledTriples(test_data)
