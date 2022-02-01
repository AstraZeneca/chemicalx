"""Tests for models."""

import inspect
import unittest
from typing import ClassVar

import torch
from class_resolver import Resolver

import chemicalx.models
from chemicalx import pipeline
from chemicalx.data import DatasetLoader, DrugComb, DrugCombDB
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
    Model,
    UnimplementedModel,
    model_resolver,
)


class TestPipeline(unittest.TestCase):
    """Test the unified training and evaluation pipeline."""

    loader: ClassVar[DatasetLoader]

    @classmethod
    def setUpClass(cls) -> None:
        """Set up the test case with a dataset."""
        cls.loader = DrugCombDB()

    def test_train_context(self):
        """Test training and evaluating on a model that uses context in its forward function."""
        model = DeepSynergy(context_channels=self.loader.context_channels, drug_channels=self.loader.drug_channels)
        results = pipeline(
            dataset=self.loader,
            model=model,
            batch_size=5120,
            epochs=1,
            context_features=True,
            drug_features=True,
            drug_molecules=False,
        )
        self.assertIsInstance(results.metrics["roc_auc"], float)

    def test_train_contextless(self):
        """Test training and evaluating on a model that does not use context in its forward function."""
        model = EPGCNDS()
        results = pipeline(
            dataset=self.loader,
            model=model,
            optimizer_kwargs=dict(lr=0.01, weight_decay=10**-7),
            batch_size=1024,
            epochs=1,
            context_features=True,
            drug_features=True,
            drug_molecules=True,
        )
        self.assertIsInstance(results.metrics["roc_auc"], float)


class MetaModelTestCase(unittest.TestCase):
    """Test model properties."""

    def test_inheritance(self):
        """Test that all models inherit from the correct class."""
        for name, model_cls in vars(chemicalx.models).items():
            if not isinstance(model_cls, type) or model_cls is Resolver:
                continue
            with self.subTest(name=name):
                self.assertTrue(
                    issubclass(model_cls, (Model, UnimplementedModel)),
                    msg=f"Model {model_cls} does not inherit from the base model class `chemicalx.models.Model`",
                )

    def test_defaults(self):
        """Test that all models have default values for all arguments."""
        skip = {"self", "drug_channels", "context_channels"}
        for model_cls in model_resolver:
            with self.subTest(name=model_cls.__name__):
                signature = inspect.signature(model_cls.__init__)
                missing = {
                    name
                    for name, param in signature.parameters.items()
                    if param.name not in skip and param.default is inspect.Parameter.empty
                }
                self.assertEqual(0, len(missing), msg=f"Missing default parameters for: {missing}")
                positional = {
                    name
                    for name, param in signature.parameters.items()
                    if param.name not in skip
                    and param.kind in {inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD}
                }
                self.assertEqual(0, len(positional), msg=f"Arguments should be kwarg only: {positional}")


class TestModels(unittest.TestCase):
    """A test case for models."""

    loader: ClassVar[DatasetLoader]

    @classmethod
    def setUpClass(cls) -> None:
        """Set up the test case with a dataset."""
        cls.loader = DrugComb()

    def setUp(self):
        """Set up the test case."""
        self.generator, _ = self.loader.get_generators(
            batch_size=32,
            context_features=True,
            drug_features=True,
            drug_molecules=True,
            train_size=0.005,
        )

    def test_caster(self):
        """Test CASTER."""
        model = CASTER(x=2)
        assert model.x == 2

    def test_epgcnds(self):
        """Test EPGCNDS."""
        model = EPGCNDS()

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
            context_channels=self.loader.context_channels,
            drug_channels=self.loader.drug_channels,
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
