import unittest

from chemicalx.data import (
    ContextFeatureSet,
    DrugFeatureSet,
    LabelSet,
)


class TestDataStructures(unittest.TestCase):
    def setUp(self):
        self.x = 2

    def test_ContextFeatureSet(self):
        data = ContextFeatureSet(x=2)
        assert data.x == 2

    def test_DrugFeatureSet(self):
        data = DrugFeatureSet(x=2)
        assert data.x == 2

    def test_LabelSet(self):
        data = LabelSet(x=2)
        assert data.x == 2
