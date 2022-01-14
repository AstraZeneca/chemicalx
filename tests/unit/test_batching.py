"""Tests for batching."""

import unittest

from chemicalx.data import BatchGenerator, DatasetLoader


class TestGeneratorDrugCombDB(unittest.TestCase):
    """Test the DrugCombDB generator."""

    def setUp(self):
        """Set up the test case."""
        loader = DatasetLoader("drugcombdb")
        self.drug_feature_set = loader.get_drug_features()
        self.context_feature_set = loader.get_context_features()
        self.labeled_triples = loader.get_labeled_triples()

    def test_all_true(self):
        """Test sizes of drug features during batch generation."""
        generator = BatchGenerator(
            batch_size=4096, context_features=True, drug_features=True, drug_molecules=True, labels=True
        )

        generator.set_data(
            context_feature_set=self.context_feature_set,
            drug_feature_set=self.drug_feature_set,
            labeled_triples=self.labeled_triples,
        )

        for batch in generator:
            assert batch.drug_features_left.shape[1] == 256
            assert (batch.drug_features_left.shape[0] == 2975) or (batch.drug_features_left.shape[0] == 4096)

    def test_set_all_false(self):
        """Test features of the batch generator."""
        generator = BatchGenerator(
            batch_size=4096, context_features=False, drug_features=False, drug_molecules=False, labels=False
        )

        generator.set_data(
            context_feature_set=self.context_feature_set,
            drug_feature_set=self.drug_feature_set,
            labeled_triples=self.labeled_triples,
        )

        for batch in generator:
            assert batch.drug_features_left is None
            assert batch.drug_molecules_left is None
            assert batch.labels is None
            assert batch.context_features is None
