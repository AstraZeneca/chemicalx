"""Datasets and utilities."""

from class_resolver import Resolver

from .contextfeatureset import ContextFeatureSet
from .datasetloader import (
    DatasetLoader,
    DrugbankDDI,
    DrugComb,
    DrugCombDB,
    RemoteDatasetLoader,
    TwoSides,
)
from .drugfeatureset import DrugFeatureSet
from .drugpairbatch import DrugPairBatch
from .labeledtriples import LabeledTriples

__all__ = [
    "ContextFeatureSet",
    "DrugFeatureSet",
    "DrugPairBatch",
    "LabeledTriples",
    # Datasets
    "dataset_resolver",
    "DatasetLoader",
    "DrugbankDDI",
    "TwoSides",
    "DrugComb",
    "DrugCombDB",
]

dataset_resolver = Resolver.from_subclasses(base=DatasetLoader, skip={RemoteDatasetLoader})
