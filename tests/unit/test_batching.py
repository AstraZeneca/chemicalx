"""Tests for batching."""

import unittest
from typing import ClassVar

from chemicalx.data import DatasetLoader, DrugCombDB


class TestGeneratorDrugCombDB(unittest.TestCase):
    """Test the DrugCombDB generator."""

    loader: ClassVar[DatasetLoader]

    @classmethod
    def setUpClass(cls) -> None:
        """Set up the class with a dataset loader."""
        cls.loader = DrugCombDB()

    def test_all_true(self):
        """Test sizes of drug features during batch generation."""
        generator = self.loader.get_generator(
            batch_size=4096,
            context_features=True,
            drug_features=True,
            drug_molecules=True,
            labeled_triples=self.loader.get_labeled_triples(),
        )
        for batch in generator:
            assert batch.drug_features_left.shape[1] == 256
            assert (batch.drug_features_left.shape[0] == 2975) or (batch.drug_features_left.shape[0] == 4096)

    def test_set_all_false(self):
        """Test features of the batch generator."""
        generator = self.loader.get_generator(
            batch_size=4096,
            context_features=False,
            drug_features=False,
            drug_molecules=False,
            labeled_triples=self.loader.get_labeled_triples(),
        )
        for batch in generator:
            assert batch.drug_features_left is None
            assert batch.drug_molecules_left is None
            assert batch.labels is not None
            assert batch.context_features is None
