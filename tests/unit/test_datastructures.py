import unittest
import numpy as np
from chemicalx.data import ContextFeatureSet, DrugFeatureSet, LabelSet


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
        assert self.another_context_feature_set.get_context_feature_count() == None

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

    def test_drug_feature_set(self):
        assert 2 == 2


class TestLabelSet(unittest.TestCase):
    """
    Testing the label set methods.
    """

    def setUp(self):
        self.x = 2

    def test_LabelSet(self):
        data = LabelSet(x=self.x)
        assert data.x == 2
