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

    def test_len(self):
        assert len(self.context_feature_set) == 2

    def test_contexts_features(self):
        assert len(list(self.context_feature_set.contexts())) == 2
        assert len(list(self.context_feature_set.features())) == 2

    def test_basic_statistics(self):
        assert self.context_feature_set.get_context_count() == 2
        assert self.context_feature_set.get_context_feature_count() == 3

    def test_density(self):
        density = self.context_feature_set.get_feature_density_rate()
        assert density == (4 / 6)

    def test_update_and_delete(self):
        self.context_feature_set.update({"context_3": np.array([1.1, 2.2, 3.4])})
        assert len(self.context_feature_set) == 3
        del self.self.context_feature_set["context_3"]
        assert len(self.context_feature_set) == 2


class TestDrugFeatureSet(unittest.TestCase):
    """
    Testing the drug feature set methods.
    """

    def setUp(self):
        self.x = 2

    def test_DrugFeatureSet(self):
        data = DrugFeatureSet(x=self.x)
        assert data.x == 2


class TestLabelSet(unittest.TestCase):
    """
    Testing the label set methods.
    """

    def setUp(self):
        self.x = 2

    def test_LabelSet(self):
        data = LabelSet(x=self.x)
        assert data.x == 2
