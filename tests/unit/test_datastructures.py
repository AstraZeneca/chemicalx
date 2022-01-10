import unittest
import numpy as np
from chemicalx.data import ContextFeatureSet, DrugFeatureSet, LabelSet


class TestContextFeatureSet(unittest.TestCase):
    def setUp(self):
        self.context_feature_set = ContextFeatureSet()
        self.context_feature_set["context_1"] = np.array([0.0, 1.8, 2.1])
        self.context_feature_set["context_2"] = np.array([0, 1, 2])

    def test_get(self):
        assert self.context_feature_set["context_2"].shape == (3,)

    def test_len(self):
        assert len(self.context_feature_set) == 2

    def test_contexts_features(self):
        assert len(self.context_feature_set.contexts()) == 2
        assert len(self.context_feature_set.features()) == 2

    def test_basic_statistics(self):
        assert self.context_feature_set.get_context_count() == 2
        assert self.context_feature_set.get_context_feature_count() == 3

    def test_denstiy(self):
        feature_matrix = self.context_feature_set.get_feature_density_rate()
        assert feature_matrix.shape == (2, 3)


class TestDrugFeatureSet(unittest.TestCase):
    def setUp(self):
        self.x = 2

    def test_DrugFeatureSet(self):
        data = DrugFeatureSet(x=self.x)
        assert data.x == 2


class TestLabelSet(unittest.TestCase):
    def setUp(self):
        self.x = 2

    def test_LabelSet(self):
        data = LabelSet(x=self.x)
        assert data.x == 2
