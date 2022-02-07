"""Datasets and utilities."""

from class_resolver import Resolver

from .batchgenerator import BatchGenerator
from .contextfeatureset import ContextFeatureSet
from .datasetloader import (
    DatasetLoader,
    DrugbankDDI,
    DrugComb,
    DrugCombDB,
    LocalDatasetLoader,
    OncoPolyPharmacology,
    RemoteDatasetLoader,
    TwoSides,
)
from .drugfeatureset import DrugFeatureSet
from .drugpairbatch import DrugPairBatch
from .labeledtriples import LabeledTriples

__all__ = [
    "BatchGenerator",
    "ContextFeatureSet",
    "DrugFeatureSet",
    "DrugPairBatch",
    "LabeledTriples",
    # Abstract datasets
    "dataset_resolver",
    "DatasetLoader",
    "RemoteDatasetLoader",
    "LocalDatasetLoader",
    # Datasets
    "DrugbankDDI",
    "TwoSides",
    "DrugComb",
    "DrugCombDB",
    "OncoPolyPharmacology",
]

dataset_resolver = Resolver.from_subclasses(base=DatasetLoader, skip={RemoteDatasetLoader, LocalDatasetLoader})
