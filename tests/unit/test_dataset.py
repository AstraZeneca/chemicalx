"""Tests for datasets."""

import pathlib
import unittest
from typing import ClassVar

from chemicalx.data import (
    DatasetLoader,
    DrugbankDDI,
    DrugComb,
    DrugCombDB,
    LocalDatasetLoader,
    TwoSides,
)

HERE = pathlib.Path(__file__).parent.resolve()


class TestDatasetLoader(LocalDatasetLoader):
    """A mock dataset loader."""

    def preprocess(self):
        """Mock the preprocessing function to be no-op."""


class TestMock(unittest.TestCase):
    """A test case for the mock dataset."""

    loader: DatasetLoader

    def setUp(self) -> None:
        """Set up the test case."""
        self.loader = TestDatasetLoader(directory=HERE.joinpath("test_dataset"))

    def test_get_context_features(self):
        """Test the number of context features."""
        assert self.loader.num_contexts == 2
        assert self.loader.context_channels == 5

    def test_get_drug_features(self):
        """Test the number of drug features."""
        assert self.loader.num_drugs == 2
        assert self.loader.drug_channels == 4

    def test_get_labeled_triples(self):
        """Test the shape of the labeled triples."""
        assert self.loader.num_labeled_triples == 2
        labeled_triples = self.loader.get_labeled_triples()
        assert labeled_triples.data.shape == (2, 4)


class TestDrugComb(unittest.TestCase):
    """A test case for DrugComb."""

    loader: ClassVar[DatasetLoader]

    @classmethod
    def setUpClass(cls) -> None:
        """Set up the test case."""
        cls.loader = DrugComb()

    def test_get_context_features(self):
        """Test the number of context features."""
        context_feature_set = self.loader.get_context_features()
        assert len(context_feature_set) == 288

    def test_get_drug_features(self):
        """Test the number of drug features."""
        drug_feature_set = self.loader.get_drug_features()
        assert len(drug_feature_set) == 4146

    def test_get_labeled_triples(self):
        """Test the shape of the labeled triples."""
        labeled_triples = self.loader.get_labeled_triples()
        assert labeled_triples.data.shape == (659333, 4)


class TestDrugCombDB(unittest.TestCase):
    """A test case for DrugCombDB."""

    loader: ClassVar[DatasetLoader]

    @classmethod
    def setUpClass(cls) -> None:
        """Set up the test case."""
        cls.loader = DrugCombDB()

    def test_get_context_features(self):
        """Test the number of context features."""
        context_feature_set = self.loader.get_context_features()
        assert len(context_feature_set) == 112

    def test_get_drug_features(self):
        """Test the number of drug features."""
        drug_feature_set = self.loader.get_drug_features()
        assert len(drug_feature_set) == 2956

    def test_get_labeled_triples(self):
        """Test the shape of the labeled triples."""
        labeled_triples = self.loader.get_labeled_triples()
        assert labeled_triples.data.shape == (191391, 4)


class TestDeepDDI(unittest.TestCase):
    """A test case for DeepDDI."""

    loader: ClassVar[DatasetLoader]

    @classmethod
    def setUpClass(cls) -> None:
        """Set up the test case."""
        cls.loader = DrugbankDDI()

    def test_get_context_features(self):
        """Test the number of context features."""
        context_feature_set = self.loader.get_context_features()
        assert len(context_feature_set) == 86

    def test_get_drug_features(self):
        """Test the number of drug features."""
        drug_feature_set = self.loader.get_drug_features()
        assert len(drug_feature_set) == 1706

    def test_get_labeled_triples(self):
        """Test the shape of the labeled triples."""
        labeled_triples = self.loader.get_labeled_triples()
        assert labeled_triples.data.shape == (383616, 4)


class TestTwoSides(unittest.TestCase):
    """A test case for TwoSides."""

    loader: ClassVar[DatasetLoader]

    @classmethod
    def setUpClass(cls) -> None:
        """Set up the test case."""
        cls.loader = TwoSides()

    def test_get_context_features(self):
        """Test the number of context features."""
        context_feature_set = self.loader.get_context_features()
        assert len(context_feature_set) == 10

    def test_get_drug_features(self):
        """Test the number of drug features."""
        drug_feature_set = self.loader.get_drug_features()
        assert len(drug_feature_set) == 644

    def test_get_labeled_triples(self):
        """Test the shape of the labeled triples."""
        labeled_triples = self.loader.get_labeled_triples()
        assert labeled_triples.data.shape == (499582, 4)
