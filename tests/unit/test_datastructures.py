import unittest
import numpy as np
from chemicalx.data import ContextFeatureSet, DrugFeatureSet, LabelSet


class TestContextFeatureSet(unittest.TestCase):
    def setUp(self):
        self.context_feature_set = ContextFeatureSet()
        self.context_feature_set["context_1"] = np.array([0, 1, 2])
        self.context_feature_set["context_2"] = np.array([0, 1, 2])

    def test_ContextFeatureSet(self):
        assert self.context_feature_set["context_2"].shape == (3,)


class DrugFeatureSet(unittest.TestCase):
    def setUp(self, x):
        self.x = 2

    def test_DrugFeatureSet(self):
        data = DrugFeatureSet(x=2)
        assert data.x == 2


class LabelSet(unittest.TestCase):
    def setUp(self, x):
        self.x = 2

    def test_LabelSet(self):
        data = LabelSet(x=2)
        assert data.x == 2
