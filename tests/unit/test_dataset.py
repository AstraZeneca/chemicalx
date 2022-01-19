"""Tests for datasets."""

import unittest

from chemicalx.data import DatasetLoader


class TestDrugComb(unittest.TestCase):
    """A test case for DrugComb."""

    @classmethod
    def setUpClass(cls) -> None:
        """Set up the test case."""
        cls.dataset_loader = DatasetLoader("drugcomb")

    def test_get_context_features(self):
        """Test the number of context features."""
        context_feature_set = self.dataset_loader.get_context_features()
        assert len(context_feature_set) == 288

    def test_get_drug_features(self):
        """Test the number of drug features."""
        drug_feature_set = self.dataset_loader.get_drug_features()
        assert len(drug_feature_set) == 4146

    def test_get_labeled_triples(self):
        """Test the shape of the labeled triples."""
        labeled_triples = self.dataset_loader.get_labeled_triples()
        assert labeled_triples.data.shape == (659333, 4)


class TestDrugCombDB(unittest.TestCase):
    """A test case for DrugCombDB."""

    @classmethod
    def setUpClass(cls) -> None:
        """Set up the test case."""
        cls.dataset_loader = DatasetLoader("drugcombdb")

    def test_get_context_features(self):
        """Test the number of context features."""
        context_feature_set = self.dataset_loader.get_context_features()
        assert len(context_feature_set) == 112

    def test_get_drug_features(self):
        """Test the number of drug features."""
        drug_feature_set = self.dataset_loader.get_drug_features()
        assert len(drug_feature_set) == 2956

    def test_get_labeled_triples(self):
        """Test the shape of the labeled triples."""
        labeled_triples = self.dataset_loader.get_labeled_triples()
        assert labeled_triples.data.shape == (191391, 4)


class TestDeepDDI(unittest.TestCase):
    """A test case for DeepDDI."""

    @classmethod
    def setUpClass(cls) -> None:
        """Set up the test case."""
        cls.dataset_loader = DatasetLoader("drugbankddi")

    def test_get_context_features(self):
        """Test the number of context features."""
        context_feature_set = self.dataset_loader.get_context_features()
        assert len(context_feature_set) == 86

    def test_get_drug_features(self):
        """Test the number of drug features."""
        drug_feature_set = self.dataset_loader.get_drug_features()
        assert len(drug_feature_set) == 1706

    def test_get_labeled_triples(self):
        """Test the shape of the labeled triples."""
        labeled_triples = self.dataset_loader.get_labeled_triples()
        assert labeled_triples.data.shape == (575307, 4)


class TestTwoSides(unittest.TestCase):
    """A test case for TwoSides."""

    @classmethod
    def setUpClass(cls) -> None:
        """Set up the test case."""
        cls.dataset_loader = DatasetLoader("twosides")

    def test_get_context_features(self):
        """Test the number of context features."""
        context_feature_set = self.dataset_loader.get_context_features()
        assert len(context_feature_set) == 10

    def test_get_drug_features(self):
        """Test the number of drug features."""
        drug_feature_set = self.dataset_loader.get_drug_features()
        assert len(drug_feature_set) == 644

    def test_get_labeled_triples(self):
        """Test the shape of the labeled triples."""
        labeled_triples = self.dataset_loader.get_labeled_triples()
        assert labeled_triples.data.shape == (499582, 4)
