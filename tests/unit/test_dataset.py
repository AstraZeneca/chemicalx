import unittest

from chemicalx.data import DatasetLoader


class TestDataset(unittest.TestCase):
    def setUp(self):
        self.x = 2

    def test_BaseDataset(self):
        assert dataset.x == 2
