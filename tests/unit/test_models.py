"""Tests for models."""

import unittest

import torch

from chemicalx import pipeline
from chemicalx.data import BatchGenerator, DatasetLoader
from chemicalx.models import (
    CASTER,
    EPGCNDS,
    GCNBMP,
    MHCADDI,
    MRGNN,
    SSIDDI,
    DeepDDI,
    DeepDDS,
    DeepDrug,
    DeepSynergy,
    MatchMaker,
)


class TestPipeline(unittest.TestCase):
    """Test the unified training and evaluation pipeline."""

    def test_train_context(self):
        """Test training and evaluating on a model that uses context in its forward function."""
        model = DeepSynergy(context_channels=112, drug_channels=256)
        results = pipeline(
            dataset="drugcombdb",
            model=model,
            batch_size=5120,
            epochs=1,
            context_features=True,
            drug_features=True,
            drug_molecules=False,
            labels=True,
        )
        self.assertIsInstance(results.roc_auc, float)

    def test_train_contextless(self):
        """Test training and evaluating on a model that does not use context in its forward function."""
        model = EPGCNDS(in_channels=69)
        results = pipeline(
            dataset="drugcombdb",
            model=model,
            optimizer_kwargs=dict(lr=0.01, weight_decay=10 ** -7),
            batch_size=1024,
            epochs=1,
            context_features=True,
            drug_features=True,
            drug_molecules=True,
            labels=True,
        )
        self.assertIsInstance(results.roc_auc, float)


class TestModels(unittest.TestCase):
    """A test case for models."""

    def setUp(self):
        """Set up the test case."""
        loader = DatasetLoader("drugcomb")
        drug_feature_set = loader.get_drug_features()
        context_feature_set = loader.get_context_features()
        labeled_triples = loader.get_labeled_triples()
        labeled_triples, _ = labeled_triples.train_test_split(train_size=0.005)
        self.generator = BatchGenerator(
            batch_size=32, context_features=True, drug_features=True, drug_molecules=True, labels=True
        )
        self.generator.set_data(context_feature_set, drug_feature_set, labeled_triples)

    def test_caster(self):
        """Test CASTER."""
        model = CASTER(x=2)
        assert model.x == 2

    def test_epgcnds(self):
        """Test EPGCNDS."""
        model = EPGCNDS(in_channels=69)

        optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=0.0001)
        model.train()
        loss = torch.nn.BCELoss()
        for batch in self.generator:
            optimizer.zero_grad()
            prediction = model(batch.drug_molecules_left, batch.drug_molecules_right)
            output = loss(prediction, batch.labels)
            output.backward()
            optimizer.step()
            assert prediction.shape[0] == batch.labels.shape[0]

    def test_gcnbmp(self):
        """Test GCNBMP."""
        model = GCNBMP(x=2)
        assert model.x == 2

    def test_mhcaddi(self):
        """Test MHCADDI."""
        model = MHCADDI(x=2)
        assert model.x == 2

    def test_mrgnn(self):
        """Test MRGNN."""
        model = MRGNN(x=2)
        assert model.x == 2

    def test_ssiddi(self):
        """Test SSIDDI."""
        model = SSIDDI(x=2)
        assert model.x == 2

    def test_deepddi(self):
        """Test DeepDDI."""
        model = DeepDDI(x=2)
        assert model.x == 2

    def test_deepdrug(self):
        """Test DeepDrug."""
        model = DeepDrug(x=2)
        assert model.x == 2

    def test_deepsynergy(self):
        """Test DeepSynergy."""
        model = DeepSynergy(
            context_channels=288,
            drug_channels=256,
            input_hidden_channels=32,
            middle_hidden_channels=16,
            final_hidden_channels=16,
            dropout_rate=0.5,
        )

        optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=0.0001)
        model.train()
        loss = torch.nn.BCELoss()
        for batch in self.generator:
            optimizer.zero_grad()
            prediction = model(batch.context_features, batch.drug_features_left, batch.drug_features_right)
            output = loss(prediction, batch.labels)
            output.backward()
            optimizer.step()
            assert prediction.shape[0] == batch.labels.shape[0]

    def test_deepdds(self):
        """Test DeepDDS."""
        model = DeepDDS(x=2)
        assert model.x == 2

    def test_matchmaker(self):
        """Test MatchMaker."""
        model = MatchMaker(x=2)
        assert model.x == 2
