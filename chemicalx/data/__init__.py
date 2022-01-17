"""Datasets and utilities."""

from class_resolver import Resolver

from .batchgenerator import BatchGenerator
from .contextfeatureset import ContextFeatureSet
from .datasetloader import DatasetLoader, DrugComb, DrugCombDB
from .drugfeatureset import DrugFeatureSet
from .drugpairbatch import DrugPairBatch
from .labeledtriples import LabeledTriples

__all__ = [
    "BatchGenerator",
    "ContextFeatureSet",
    "DrugFeatureSet",
    "DrugPairBatch",
    "LabeledTriples",
    # Datasets
    "dataset_resolver",
    "DatasetLoader",
    "DrugCombDatasetLoader",
    "DrugCombDbDatasetLoader",
]

dataset_resolver = Resolver.from_subclasses(base=DatasetLoader)
