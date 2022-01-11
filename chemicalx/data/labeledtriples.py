import pandas as pd
from typing import List, Tuple
from sklearn.model_selection import train_test_split


class LabeledTriples:
    def __init__(self):
        self.columns = ["drug_1", "drug_2", "context", "label"]
        self.types = {"drug_1": str, "drug_2": str, "context": str, "label": float}
        self.data = pd.DataFrame(columns=self.columns).astype(self.types)

    def drop_duplicates(self):
        self.data = self.data.drop_duplicates()

    def update_from_pandas(self, data: pd.DataFrame):
        self.data = pd.concat([self.data, data])

    def update_from_list(self, data: List[List]):
        data = pd.DataFrame(data, columns=self.columns)
        self.data = pd.concat([self.data, data])

    def __add__(self, value):
        new_triples = LabeledTriples()
        new_triples.update_from_pandas(pd.concat([self.data, value.data]))
        return new_triples

    def get_drug_count(self) -> int:
        return pd.unique(self.data[["drug_1", "drug_2"]].values.ravel("K")).shape[0]

    def get_context_count(self) -> int:
        return self.data["context"].nunique()

    def get_combination_count(self) -> int:
        combination_count = self.data[["drug_1", "drug_2"]].drop_duplicates().shape[0]
        return combination_count

    def get_labeled_triple_count(self) -> int:
        triple_count = self.data.shape[0]
        return triple_count

    def get_positive_count(self) -> int:
        return int(self.data["label"].sum())

    def get_negative_count(self) -> int:
        return self.get_labeled_triple_count() - self.get_positive_count()

    def get_positive_rate(self) -> float:
        return self.data["label"].mean()

    def get_negative_rate(self) -> float:
        return 1.0 - self.data["label"].mean()

    def train_test_split(self, train_size: float = 0.8, random_state: int = 42) -> Tuple:

        train_data, test_data = train_test_split(self.data, train_size=train_size, random_state=random_state)

        train_labeled_triples = LabeledTriples()
        test_labeled_triples = LabeledTriples()

        train_labeled_triples.update_from_pandas(train_data)
        test_labeled_triples.update_from_pandas(test_data)

        return train_labeled_triples, test_labeled_triples
