import unittest
import numpy as np
import pandas as pd
from chemicalx.data import ContextFeatureSet, DrugFeatureSet, LabeledTriples


class TestContextFeatureSet(unittest.TestCase):
    """
    Testing the context feature set methods.
    """

    def setUp(self):
        self.context_feature_set = ContextFeatureSet()
        self.context_feature_set["context_1"] = np.array([0.0, 1.8, 2.1])
        self.context_feature_set["context_2"] = np.array([0, 1, 2])

    def test_get(self):
        assert self.context_feature_set["context_2"].shape == (1, 3)
        assert ("context_2" in self.context_feature_set) == True
        assert self.context_feature_set.has_context("context_2") == True

    def test_delete(self):
        self.another_context_feature_set = self.context_feature_set
        del self.another_context_feature_set["context_1"]
        del self.another_context_feature_set["context_2"]
        assert self.another_context_feature_set.get_context_feature_count() == 0

    def test_len(self):
        assert len(self.context_feature_set) == 2

    def test_contexts_features(self):
        assert len(list(self.context_feature_set.keys())) == 2
        assert len(list(self.context_feature_set.values())) == 2
        assert len(list(self.context_feature_set.items())) == 2

    def test_basic_statistics(self):
        assert self.context_feature_set.get_context_count() == 2
        assert self.context_feature_set.get_context_feature_count() == 3

    def test_density(self):
        density = self.context_feature_set.get_feature_density_rate()
        assert density == (4 / 6)

    def test_update_and_delete(self):
        self.context_feature_set.update({"context_3": np.array([1.1, 2.2, 3.4])})
        assert len(self.context_feature_set) == 3
        del self.context_feature_set["context_3"]
        assert len(self.context_feature_set) == 2

    def test_iteration(self):
        for context in self.context_feature_set:
            feature_vector = self.context_feature_set[context]
            assert feature_vector.shape == (1, 3)

    def test_clearing(self):
        self.context_feature_set.clear()
        assert len(self.context_feature_set) == 0


class TestDrugFeatureSet(unittest.TestCase):
    """
    Testing the drug feature set methods.
    """

    def setUp(self):
        self.drug_feature_set = DrugFeatureSet()
        self.drug_feature_set["drug_1"] = {"smiles": "CN=C=O", "features": np.array([0.0, 1.7, 2.3])}
        self.drug_feature_set["drug_2"] = {"smiles": "[Cu+2].[O-]S(=O)(=O)[O-]", "features": np.array([1, 0, 8])}

    def test_get(self):
        assert self.drug_feature_set["drug_1"]["features"].shape == (1, 3)
        assert len(self.drug_feature_set["drug_1"]["smiles"]) == 6
        assert ("drug_2" in self.drug_feature_set) == True
        assert self.drug_feature_set.has_drug("drug_2") == True

    def test_delete(self):
        self.another_drug_feature_set = self.drug_feature_set
        del self.another_drug_feature_set["drug_1"]
        del self.another_drug_feature_set["drug_2"]
        assert self.another_drug_feature_set.get_drug_feature_count() == 0

    def test_len(self):
        assert len(self.drug_feature_set) == 2

    def test_drug_features(self):
        assert len(list(self.drug_feature_set.keys())) == 2
        assert len(list(self.drug_feature_set.values())) == 2
        assert len(list(self.drug_feature_set.items())) == 2

    def test_basic_statistics(self):
        assert self.drug_feature_set.get_drug_count() == 2
        assert self.drug_feature_set.get_drug_feature_count() == 3

    def test_density(self):
        density = self.drug_feature_set.get_feature_density_rate()
        assert density == (4 / 6)

    def test_update_and_delete(self):
        self.drug_feature_set.update(
            {"drug_3": {"smiles": " CN1C=NC2=C1C(=O)N(C(=O)N2C)C", "features": np.array([1.1, 2.2, 3.4])}}
        )
        assert len(self.drug_feature_set) == 3
        del self.drug_feature_set["drug_3"]
        assert len(self.drug_feature_set) == 2

    def test_iteration(self):
        for drug in self.drug_feature_set:
            features = self.drug_feature_set[drug]
            assert len(features) == 2

    def test_clearing(self):
        self.drug_feature_set.clear()
        assert len(self.drug_feature_set) == 0

    def test_get_smiles(self):
        smiles_strings = self.drug_feature_set.get_smiles_strings(list(self.drug_feature_set.keys()))
        assert len(smiles_strings) == 2


class TestLabeledTriples(unittest.TestCase):
    """
    Testing the labeled triples methods.
    """

    def setUp(self):
        self.labeled_triples = LabeledTriples()
        self.other_labeled_triples = LabeledTriples()

        data = pd.DataFrame(
            [["drug_a", "drug_b", "context_a", 1.0], ["drug_b", "drug_c", "context_b", 0.0]],
            columns=["drug_1", "drug_2", "context", "label"],
        )
        self.labeled_triples.update_from_pandas(data)

        data = [["drug_a", "drug_b", "context_a", 1.0], ["drug_a", "drug_c", "context_b", 0.0]]
        self.other_labeled_triples.update_from_list(data)

    def test_from_pandas(self):
        assert self.labeled_triples.data.shape == (2, 4)

    def test_from_list(self):
        assert self.other_labeled_triples.data.shape == (2, 4)

    def test_add_and_drops(self):
        labeled_triples = self.other_labeled_triples + self.labeled_triples
        assert labeled_triples.data.shape == (4, 4)
        labeled_triples.drop_duplicates()
        assert labeled_triples.data.shape == (3, 4)

    def test_split(self):
        labeled_triples = self.other_labeled_triples + self.labeled_triples
        train_triples, test_triples = labeled_triples.train_test_split(train_size=0.5, random_state=42)

        assert train_triples.data.shape == (2, 4)
        assert test_triples.data.shape == (2, 4)

    def test_counts(self):
        labeled_triples = self.other_labeled_triples + self.labeled_triples
        train_triples, test_triples = labeled_triples.train_test_split(train_size=0.5, random_state=42)
