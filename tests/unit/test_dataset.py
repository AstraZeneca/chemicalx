import unittest
from chemicalx.data import DatasetLoader


class TestDrugComb(unittest.TestCase):
    def setUp(self):
        self.dataset_loader = DatasetLoader("drugcomb")

    def test_BaseDataset(self):
        assert 2 == 2


class TestDrugCombDB(unittest.TestCase):
    def setUp(self):
        self.dataset_loader = DatasetLoader("drugcombdb")

    def test_BaseDataset(self):
        assert 2 == 2
