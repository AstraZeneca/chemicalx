import unittest

from chemicalx.data import DatasetLoader


class TestDrugComb(unittest.TestCase):
    def setUp(self):
        self.dataset_loader = DatasetLoader("drugcomb")

    def test_get_context_features(self):
        context_feature_set = self.dataset_loader.get_context_features()
        assert len(context_feature_set) == 288

    def test_get_drug_features(self):
        drug_feature_set = self.dataset_loader.get_drug_features()
        assert len(drug_feature_set) == 4146

    def test_get_labeled_triples(self):
        labeled_triples = self.dataset_loader.get_labeled_triples()
        assert labeled_triples.data.shape == (659333, 4)


class TestDrugCombDB(unittest.TestCase):
    def setUp(self):
        self.dataset_loader = DatasetLoader("drugcombdb")

    def test_get_context_features(self):
        context_feature_set = self.dataset_loader.get_context_features()
        assert len(context_feature_set) == 112

    def test_get_drug_features(self):
        drug_feature_set = self.dataset_loader.get_drug_features()
        assert len(drug_feature_set) == 2956

    def test_get_labeled_triples(self):
        labeled_triples = self.dataset_loader.get_labeled_triples()
        assert labeled_triples.data.shape == (191391, 4)
